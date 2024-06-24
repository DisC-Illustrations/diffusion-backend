from enum import Enum
import numpy as np
from PIL import Image
from diffusers import DiffusionPipeline, StableDiffusionUpscalePipeline, StableDiffusionLatentUpscalePipeline
from diffusers import AutoencoderKL
import torch


class StableDiffusionModel(Enum):
    STABLE_DIFFUSION_1_5 = "runwayml/stable-diffusion-v1-5"
    STABLE_DIFFUSION_XL = "stabilityai/stable-diffusion-xl-base-1.0"
    STABLE_DIFFUSION_3_MEDIUM = "stabilityai/stable-diffusion-3-medium"


class Upscaler(Enum):
    X2 = "stabilityai/sd-x2-latent-upscaler"
    X4 = "stabilityai/stable-diffusion-x4-upscaler"


def upscale_images(images, model: Upscaler, pipeline: DiffusionPipeline, num_images: int, prompt: str):
    if isinstance(images, list):
        images = np.array(images)

    high_res_images = []
    upscaler = None
    if model == Upscaler.X2:
        upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(model.value, torch_dtype=torch.float16)
    elif model == Upscaler.X4:
        upscaler = StableDiffusionUpscalePipeline.from_pretrained(model.value, torch_dtype=torch.float16)

    upscaler.to(pipeline.device)
    upscaler.enable_attention_slicing()

    for i, image in enumerate(images):
        high_res_image = upscaler(image=pipeline.numpy_to_pil(np.array(image)), num_inference_steps=20,
                                  prompt=prompt, output_type="pil").images[0]
        print(f"Upscaled image {i + 1}: Size={high_res_image.size}, Mode={high_res_image.mode}")
        high_res_images.append(high_res_image)
        torch.cuda.empty_cache()  # Leere den CUDA-Cache nach jedem Bild

    return high_res_images


def initialize_pipeline(model_id: str):
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"

    dtype = torch.float32
    if device == "cuda":
        dtype = torch.float16

    pipeline = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype, use_safetensors=True).to(device)
    if model_id == StableDiffusionModel.STABLE_DIFFUSION_XL.value and device == "cuda":
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        pipeline.vae = vae
    pipeline.enable_attention_slicing()
    pipeline.enable_vae_slicing()
    return pipeline


class Diffuser:
    def __init__(self, model_id: str = StableDiffusionModel.STABLE_DIFFUSION_XL.value):
        self.model_id = model_id
        self.pipeline = initialize_pipeline(model_id)

    @classmethod
    def from_model(cls, model: StableDiffusionModel):
        return cls(model.value)

    def generate_image(self, prompt, negative_prompt="text, watermarks",
                       num_images=1, width=512, height=512, steps=25, upscale=1):
        images = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_images_per_prompt=num_images,
            num_inference_steps=steps
        ).images

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

        if upscale in [2, 4]:
            upscaler = Upscaler.X2 if upscale == 2 else Upscaler.X4
            high_res_images = upscale_images(images, upscaler, self.pipeline, num_images, prompt)
        else:
            high_res_images = images

        # high_res_images = [process_image(img) for img in high_res_images]

        return high_res_images
