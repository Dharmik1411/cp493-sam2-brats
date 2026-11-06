"""SAM 2.1 on one MRI volume: a box on every slice, or boxes on a few slices + propagation."""
import os
import tempfile

import numpy as np
from PIL import Image

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # Mac: run ops MPS doesn't have on the CPU
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # Apple Silicon GPU
        return "mps"
    return "cpu"


class Sam2Runner:
    def __init__(self, model_id="facebook/sam2.1-hiera-small", lora_path=None):
        device = pick_device()
        print("SAM 2 on", device)
        self.image = SAM2ImagePredictor.from_pretrained(model_id, device=device)
        self.video = SAM2VideoPredictor.from_pretrained(model_id, device=device)
        if lora_path:  # H4: adapters trained by train_lora.py
            from lora import add_lora

            ck = torch.load(lora_path, map_location="cpu")
            for m in (self.image.model, self.video):
                add_lora(m, ck["part"], ck["rank"])
                bad = m.load_state_dict(ck["weights"], strict=False).unexpected_keys
                assert not bad, f"LoRA weights do not fit this model: {bad[:3]}"
                m.eval()
            print(f"LoRA on {ck['part']}, rank {ck['rank']}, from {lora_path}")

    @torch.inference_mode()
    def per_slice(self, frames, boxes):
        """boxes = {slice: box}. Each slice is segmented on its own."""
        out = np.zeros(frames.shape[:3], bool)
        for z, box in boxes.items():
            self.image.set_image(frames[z])
            masks, _, _ = self.image.predict(box=box, multimask_output=False)
            out[z] = masks[0] > 0
        return out

    def load_volume(self, frames):
        # the video predictor reads a folder of JPEGs named 00000.jpg, 00001.jpg, ...
        with tempfile.TemporaryDirectory() as tmp:
            for z in range(len(frames)):
                Image.fromarray(frames[z]).save(f"{tmp}/{z:05d}.jpg", quality=95)
            return self.video.init_state(video_path=tmp)

    @torch.inference_mode()
    def propagate(self, state, boxes, direction="both"):
        """boxes = {slice: box} on the key slices. direction = 'forward' or 'both'."""
        fwd = self._pass(state, boxes, reverse=False)
        if direction == "forward":
            return fwd
        rev = self._pass(state, boxes, reverse=True)
        # for each slice keep the pass whose prompted slice is closer
        keys = sorted(boxes)
        out = fwd.copy()
        for z in range(len(out)):
            below = [k for k in keys if k <= z]
            above = [k for k in keys if k >= z]
            d_fwd = z - below[-1] if below else np.inf
            d_rev = above[0] - z if above else np.inf
            if d_rev < d_fwd:
                out[z] = rev[z]
        return out

    @torch.inference_mode()
    def reprompt(self, state, boxes, plan):
        """Dr. Zia's protocol: each (start, last, reverse) segment from brats.reprompt_plan runs on its
        own, with the memory reset and one box on `start`. The seed is kept from the forward pass."""
        out = np.zeros((state["num_frames"], state["video_height"], state["video_width"]), bool)
        done = np.zeros(state["num_frames"], bool)
        for start, last, reverse in plan:
            self.video.reset_state(state)
            self.video.add_new_points_or_box(state, frame_idx=start, obj_id=1, box=boxes[start])
            for z, _, logits in self.video.propagate_in_video(
                    state, start_frame_idx=start, max_frame_num_to_track=abs(last - start), reverse=reverse):
                if not done[z]:
                    out[z], done[z] = (logits[0, 0] > 0).cpu().numpy(), True
        return out

    def _pass(self, state, boxes, reverse):
        self.video.reset_state(state)  # each pass starts only from the boxes
        for z, box in boxes.items():
            self.video.add_new_points_or_box(state, frame_idx=z, obj_id=1, box=box)
        out = np.zeros((state["num_frames"], state["video_height"], state["video_width"]), bool)
        start = max(boxes) if reverse else min(boxes)
        for z, _, logits in self.video.propagate_in_video(state, start_frame_idx=start, reverse=reverse):
            out[z] = (logits[0, 0] > 0).cpu().numpy()
        return out
