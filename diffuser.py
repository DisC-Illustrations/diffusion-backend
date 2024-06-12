from enum import Enum

from diffusers import StableDiffusionPipeline, StableDiffusionUpscalePipeline, StableDiffusionLatentUpscalePipeline


class StableDiffusionModel(Enum):
    STABLE_DIFFUSION_1_5 = "runwayml/stable-diffusion-v1-5"
    STABLE_DIFFUSION_2_1 = "stabilityai/stable-diffusion-2-1"
    STABLE_DIFFUSION_1_4 = "CompVis/stable-diffusion-v1-4"
    STABLE_DIFFUSION_XL = "stabilityai/stable-diffusion-xl-base-1.0"
    STABLE_DIFFUSION_3_MEDIUM = "stabilityai/stable-diffusion-3-medium"


class Upscaler(Enum):
    X2 = "stabilityai/sd-x2-latent-upscaler"
    X4 = "stabilityai/stable-diffusion-x4-upscaler"


upscalers = {}


def get_upscaler(model: Upscaler):
    if model not in upscalers:
        if model == Upscaler.X2:
            upscalers[model] = StableDiffusionLatentUpscalePipeline(model.value)
        elif model == Upscaler.X4:
            upscalers[model] = StableDiffusionUpscalePipeline(model.value)

    return upscalers[model]


def initialize_pipeline(model_id: str):
    return StableDiffusionPipeline.from_pretrained(model_id)


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
            num_images=num_images,
            num_inference_steps=steps)["images"]

        # apply upscaling
        if upscale == 2:
            scaler = get_upscaler(Upscaler.X2)
            high_res_images = scaler(images, num_images_per_prompt=num_images)["images"]
        elif upscale == 4:
            scaler = get_upscaler(Upscaler.X4)
            high_res_images = scaler(images, num_images_per_prompt=num_images)["images"]
        else:
            high_res_images = images

        return high_res_images
