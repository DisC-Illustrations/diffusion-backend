from enum import Enum

from diffusers import DiffusionPipeline, StableDiffusionUpscalePipeline, StableDiffusionLatentUpscalePipeline
import torch


class StableDiffusionModel(Enum):
    STABLE_DIFFUSION_1_5 = "runwayml/stable-diffusion-v1-5"
    STABLE_DIFFUSION_XL = "stabilityai/stable-diffusion-xl-base-1.0"
    STABLE_DIFFUSION_3_MEDIUM = "stabilityai/stable-diffusion-3-medium"


class Upscaler(Enum):
    X2 = "stabilityai/sd-x2-latent-upscaler"
    X4 = "stabilityai/stable-diffusion-x4-upscaler"


def upscale_images(images, model: Upscaler, pipeline: DiffusionPipeline, num_images: int, prompt: str):
    high_res_images = None
    if model == Upscaler.X2:
        upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(model.value, torch_dtype=pipeline.torch_dtype)
        upscaler.to(pipeline.device)
        high_res_images = upscaler(image=pipeline.numpy_to_pil(images), num_inference_steps=20,
                                   num_images_per_prompt=num_images, prompt=prompt, output_type="pil").images
    elif model == Upscaler.X4:
        upscaler = StableDiffusionUpscalePipeline.from_pretrained(model.value, torch_dtype=pipeline.torch_dtype)
        upscaler.to(pipeline.device)
        high_res_images = upscaler(image=pipeline.numpy_to_pil(images), num_inference_steps=20,
                                   output_type="pil").images

    return high_res_images


def initialize_pipeline(model_id: str):
    device = "cpu"

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"

    return DiffusionPipeline.from_pretrained(model_id, use_safetensors=True).to(device)


class Diffuser:
    def __init__(self, model_id: str = StableDiffusionModel.STABLE_DIFFUSION_XL.value):
        self.model_id = model_id
        self.pipeline = initialize_pipeline(model_id)

    @classmethod
    def from_model(cls, model: StableDiffusionModel):
        instance = cls(model.value)
        instance.pipeline = initialize_pipeline(model.value)
        return instance

    def generate_image(self, prompt, negative_prompt="text, watermarks",
                       num_images=1, width=1024, height=1024, steps=25, upscale=1):

        images = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_images_per_prompt=num_images,
            num_inference_steps=steps).images

        # apply upscaling
        if upscale == 2:
            high_res_images = (
                upscale_images(images, Upscaler.X2, self.pipeline, num_images, prompt))
        elif upscale == 4:
            high_res_images = (
                upscale_images(images, Upscaler.X4, self.pipeline, num_images, prompt))
        else:
            high_res_images = images

        return high_res_images
