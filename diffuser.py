from enum import Enum

import numpy as np
import torch
from PIL import Image, ImageOps
from diffusers import DiffusionPipeline, StableDiffusionUpscalePipeline, StableDiffusionLatentUpscalePipeline


class StableDiffusionModel(Enum):
    STABLE_DIFFUSION_1_5 = "runwayml/stable-diffusion-v1-5"
    STABLE_DIFFUSION_XL = "stabilityai/stable-diffusion-xl-base-1.0"
    STABLE_DIFFUSION_3_MEDIUM = "stabilityai/stable-diffusion-3-medium"


class Upscaler(Enum):
    X2 = "stabilityai/sd-x2-latent-upscaler"
    X4 = "stabilityai/stable-diffusion-x4-upscaler"


def clear_caches():
    # clear CUDA if it is available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("Cleared CUDA cache")
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
        print("Cleared MPS cache")
    else:
        print("No cache to clear")


def upscale_images(images, model: Upscaler, pipeline: DiffusionPipeline, prompt: str):
    if isinstance(images, list):
        images = np.array(images)

    dtype = torch.float32
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.float16
    print(f"Upscaling {len(images)} images with {model.value} using {dtype} dtype")
    high_res_images = []
    upscaler = None
    if model == Upscaler.X2:
        upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(model.value, torch_dtype=dtype)
    elif model == Upscaler.X4:
        upscaler = StableDiffusionUpscalePipeline.from_pretrained(model.value, torch_dtype=dtype)

    upscaler.to(pipeline.device)
    upscaler.enable_attention_slicing()

    for i, image in enumerate(images):
        high_res_image = upscaler(image=pipeline.numpy_to_pil(np.array(image)), num_inference_steps=20,
                                  prompt=prompt, output_type="pil").images[0]
        print(f"Upscaled image {i + 1}: Size={high_res_image.size}, Mode={high_res_image.mode}")
        high_res_images.append(high_res_image)
        clear_caches()

    return high_res_images


def initialize_pipeline(model_id: str):
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"

    pipeline = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32, use_safetensors=True).to(device)
    pipeline.enable_attention_slicing()
    pipeline.enable_vae_slicing()
    return pipeline


def apply_palette(img: Image.Image, palette: list) -> Image.Image:
    print(palette)
    if not palette:  # Handle empty or None palette
        return img

    flat_palette = [value for color in palette for value in color]

    palette_image = img.convert("P", palette=Image.ADAPTIVE, colors=len(palette))
    palette_image.putpalette(flat_palette)

    return palette_image


def process_image(img, palette, is_upscaled: bool = False):
    # invert image because of bug in CUDA implementation of the upscalers
    if is_upscaled and torch.cuda.is_available():
        if isinstance(img, Image.Image):
            img = ImageOps.invert(img.convert("RGB"))
        else:
            img = ImageOps.invert(Image.fromarray(img))

    img = apply_palette(img, palette)

    return img


class Diffuser:
    def __init__(self, model_id: str = StableDiffusionModel.STABLE_DIFFUSION_XL.value):
        self.model_id = model_id
        self.pipeline = initialize_pipeline(model_id)

    @classmethod
    def from_model(cls, model: StableDiffusionModel):
        return cls(model.value)

    def generate_image(self, prompt, color_palette=None, negative_prompt="text, watermarks",
                       num_images=1, width=512, height=512, steps=25, upscale=1):
        images = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_images_per_prompt=num_images,
            num_inference_steps=steps
        ).images

        clear_caches()

        for i, img in enumerate(images):
            img_array = np.array(img)
            print(f"Image {i + 1} stats:")
            print(f"  Min value: {np.min(img_array)}")
            print(f"  Max value: {np.max(img_array)}")
            print(f"  Mean value: {np.mean(img_array)}")
            print(f"  Has NaN: {np.isnan(img_array).any()}")
            print(f"  Has Inf: {np.isinf(img_array).any()}")

        print(f"Generated images: {len(images)}")
        for i, img in enumerate(images):
            print(f"Image {i + 1}: Size={img.size}, Mode={img.mode}")

        is_upscaled: bool = False
        if upscale in [2, 4]:
            upscaler = Upscaler.X2 if upscale == 2 else Upscaler.X4
            high_res_images = upscale_images(images, upscaler, self.pipeline, prompt)
            is_upscaled = True
        else:
            high_res_images = images

        high_res_images = [process_image(img, color_palette, is_upscaled) for img in high_res_images]

        clear_caches()
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()

        return high_res_images
