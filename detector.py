"""Zero-shot violence detection in images and video with CLIP."""

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"

# several phrasings per class; their text embeddings are averaged (prompt ensembling, Radford et al. 2021)
PROMPTS = {
    "violent": ["a photo of people fighting", "a photo of a street fight", "a photo of someone punching a person",
                "a photo of a violent brawl", "a CCTV image of people fighting", "a photo of a person being attacked"],
    "non-violent": ["a photo of people walking", "a photo of people talking", "a photo of people playing sports",
                    "a photo of people eating together", "a CCTV image of a calm street", "a photo of a peaceful crowd"],
}
SINGLE_PROMPT = {"violent": ["a photo of violence"], "non-violent": ["a photo of people"]}   # baseline


class ViolenceDetector:
    def __init__(self, model_name=MODEL_NAME, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.class_embeddings = self.embed_prompts(PROMPTS)

    @torch.no_grad()
    def embed_images(self, images, batch_size=32):
        """unit-length CLIP embeddings for a list of RGB PIL images or arrays"""
        out = []
        for start in range(0, len(images), batch_size):
            inputs = self.processor(images=images[start:start + batch_size], return_tensors="pt").to(self.device)
            features = self.model.get_image_features(**inputs)
            out.append(torch.nn.functional.normalize(features, dim=-1).cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def embed_prompts(self, prompts):
        """one unit vector per class: the normalised mean of its prompts' embeddings"""
        rows = []
        for texts in prompts.values():
            inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
            features = torch.nn.functional.normalize(self.model.get_text_features(**inputs), dim=-1).mean(dim=0)
            rows.append(torch.nn.functional.normalize(features, dim=-1).cpu().numpy())
        return np.stack(rows)

    def violence_probability(self, image_embeddings, class_embeddings=None):
        """softmax over the two classes, using CLIP's learned temperature"""
        class_embeddings = self.class_embeddings if class_embeddings is None else class_embeddings
        logits = self.model.logit_scale.exp().item() * image_embeddings @ class_embeddings.T
        return softmax(logits)[:, 0]

    def predict_image(self, image):
        return float(self.violence_probability(self.embed_images([image]))[0])

    def predict_video(self, path, every_s=0.5, window=5):
        """per-frame probabilities, smoothed over `window` sampled frames, plus their timestamps"""
        frames, times = sample_frames(path, every_s)
        probs = self.violence_probability(self.embed_images(frames))
        return times, smooth(probs, window)


def softmax(logits):
    z = np.exp(logits - logits.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


def smooth(values, window):
    """centred moving average; a single violent-looking frame should not flip the whole segment"""
    if window <= 1 or len(values) < 2:
        return np.asarray(values, dtype=float)
    padded = np.pad(np.asarray(values, dtype=float), window // 2, mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")[:len(values)]


def frame_indices(n_frames, fps, every_s):
    step = max(1, round(fps * every_s))
    return list(range(0, n_frames, step))


def even_indices(n_frames, k):
    """k frames spread evenly across the clip, avoiding the very first and last frame"""
    return np.linspace(0, n_frames - 1, k + 2)[1:-1].round().astype(int).tolist()


def read_frames(path, indices):
    """RGB PIL frames at the given indices and the indices that decoded (OpenCV gives BGR; CLIP expects RGB)"""
    capture = cv2.VideoCapture(path)
    frames, kept = [], []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if ok:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            kept.append(index)
    capture.release()
    return frames, kept


def sample_frames(path, every_s=0.5):
    capture = cv2.VideoCapture(path)
    n_frames, fps = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), capture.get(cv2.CAP_PROP_FPS) or 25.0
    capture.release()
    frames, kept = read_frames(path, frame_indices(n_frames, fps, every_s))
    return frames, np.array(kept) / fps
