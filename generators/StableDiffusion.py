import time
from typing import Optional

import torch
from diffusers import DiffusionPipeline


class StableDiffusion:
    def __init__(self, model_id: str, device: str = "cpu"):
        self.pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32, use_safetensors=True)
        self.pipe.to(device)
        self.pipe.enable_attention_slicing()
        self.pipe.enable_vae_slicing()

    def generate_images(self, prompt: str, negative_prompt: str, width: int = 1024, height: int = 1024,
                        num_inference_steps: int = 30, num_images: int = 1):
        print(f"Generating images with {self.pipe.name_or_path}...")
        start_time = time.time()

        images = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_images_per_prompt=num_images,
            num_inference_steps=num_inference_steps,
            output_type="pil"
        ).images

        print(f"Generated {len(images)} images in {time.time() - start_time} seconds")
        return images
