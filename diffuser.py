from enum import Enum

from diffusers import StableDiffusionPipeline, StableDiffusionUpscalePipeline, StableDiffusionLatentUpscalePipeline


class StableDiffusionModel(Enum):
    STABLE_DIFFUSION_1_5 = "runwayml/stable-diffusion-v1-5"
    STABLE_DIFFUSION_2_1 = "stabilityai/stable-diffusion-2-1"
    STABLE_DIFFUSION_1_4 = "CompVis/stable-diffusion-v1-4"
    STABLE_DIFFUSION_XL = "stabilityai/stable-diffusion-xl-base-1.0"


class Upscaler(Enum):
    NO_UPSCALE = ""
    X2 = "stabilityai/sd-x2-latent-upscaler"
    X4 = "stabilityai/stable-diffusion-x4-upscaler"


class Diffuser:
    def __init__(self, model_id: str = StableDiffusionModel.STABLE_DIFFUSION_XL.value):
        self.model_id = model_id
        self.pipeline = self.initialize_pipeline(model_id)

    @classmethod
    def from_model(cls, model: StableDiffusionModel):
        instance = cls(model.value)
        instance.pipeline = instance.initialize_pipeline(model.value)
        return instance

    def initialize_pipeline(self, model_id: str):
        return StableDiffusionPipeline.from_pretrained(model_id)

    def generate_image(self, prompt, num_images=1, width=1024, height=1024, steps=25, upscale=0):
        images = self.pipeline(
            prompt,
            width=width,
            height=height,
            num_images=num_images,
            num_inference_steps=steps)["images"]

        if upscale is 2:
            scaler = StableDiffusionLatentUpscalePipeline.from_pretrained(Upscaler.X2.value)
            high_res_images = scaler(images, num_images_per_prompt=num_images)["images"]
        elif upscale is 4:
            scaler = StableDiffusionUpscalePipeline.from_pretrained(Upscaler.X4.value)
            high_res_images = scaler(images, num_images_per_prompt=num_images)["images"]
        else:
            high_res_images = images

        return images
