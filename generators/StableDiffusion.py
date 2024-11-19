import time

import torch
from diffusers import DiffusionPipeline, BitsAndBytesConfig
from transformers import T5Tokenizer, T5EncoderModel

from models import DiffusionModel


class StableDiffusion:
    def __init__(self, model_id: str, device: str = "cpu"):
        # float16 if CUDA is available, otherwise float32
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype, use_safetensors=True)

        # set Text Encoder to T5_xxl for SD 3.5
        if model_id == DiffusionModel.STABLE_DIFFUSION_3_5_MEDIUM.value:
            t5_encoder = T5EncoderModel.from_pretrained(
                "stabilityai/stable-diffusion-3.5-medium",
                subfolder="text_encoder_3",
                device_map="auto",
            )

            # Tokenizer laden
            t5_tokenizer = T5Tokenizer.from_pretrained(
                "stabilityai/stable-diffusion-3.5-medium",
                subfolder="tokenizer_3",
            )

            self.pipe.tokenizer = t5_tokenizer
            self.pipe.text_encoder = t5_encoder

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
