import time
import os
from typing import Optional

import torch
from diffusers import FluxTransformer2DModel, FluxPipeline, DiffusionPipeline
from optimum.quanto import freeze, qfloat8, quantize
from transformers import T5EncoderModel

pipe: FluxPipeline | None = None
bfl_repo = "black-forest-labs/FLUX.1-schnell"
dtype = torch.float16


def init_pipeline(device: str):
    print("Initializing Flux Pipeline. This may take a few minutes...")
    start_time = time.time()
    transformer = FluxTransformer2DModel.from_single_file(
        "https://huggingface.co/Kijai/flux-fp8/blob/main/flux1-schnell-fp8-e4m3fn.safetensors",
        torch_dtype=dtype, token=os.getenv("HF_TOKEN"))
    quantize(transformer, qfloat8)
    freeze(transformer)

    text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype)
    quantize(text_encoder_2, qfloat8)
    freeze(text_encoder_2)

    global pipe
    pipe = FluxPipeline.from_pretrained(bfl_repo, transformer=None, text_encoder_2=None, torch_dtype=dtype)
    pipe.transformer = transformer
    pipe.text_encoder_2 = text_encoder_2

    print(f"Using device: {device}")
    pipe.to(device)
    if torch.cuda.is_available():
        pipe.enable_model_cpu_offload()

    print(f"Initialized Flux Pipeline in {time.time() - start_time} seconds")


def generate_images(prompt: str, width: int = 1024, height: int = 1024, num_inference_steps: int = 30,
                    num_variations: int = 1, device: str = "cpu"):
    global pipe
    if pipe is None:
        init_pipeline(device)

    if pipe is None:
        raise RuntimeError("Flux Pipeline is not initialized")

    print(f"Generating images with {bfl_repo}...")

    prompt = prompt + ", no watermarks, no text"
    current_time = time.time()
    images = pipe(
        prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=num_variations,
        guidance_scale=3.5,
        output_type="pil",
    ).images

    print(f"Generated {len(images)} images in {time.time() - current_time} seconds")
    return images
