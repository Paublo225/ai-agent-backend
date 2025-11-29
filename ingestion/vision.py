"""Self-hosted vision model wrapper (LLaVA-NeXT) for diagrams."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration


class LocalVisionAnalyzer:
    def __init__(self, model_name: str = "llava-hf/llava-v1.6-34b-hf") -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("LocalVisionAnalyzer requires a CUDA-enabled GPU.")
        self.device = "cuda"
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )

    def analyze(self, image_path: Path) -> Dict[str, str]:
        image = Image.open(image_path).convert("RGB")
        prompt = (
            "You are an appliance service technician. Describe the diagram, "
            "list part numbers, and identify appliance type and model labels."
        )
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        generated = self.model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.2,
        )
        text = self.processor.decode(generated[0], skip_special_tokens=True)
        return {"description": text}
