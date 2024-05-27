from enum import Enum
from diffusers import StableDiffusionPipeline
import torch


class StableDiffusionModel(Enum):
    STABLE_DIFFUSION_1_5 = "runwayml/stable-diffusion-v1-5"
    STABLE_DIFFUSION_2_1 = "stabilityai/stable-diffusion-2-1"
    STABLE_DIFFUSION_1_4 = "CompVis/stable-diffusion-v1-4"
    STABLE_DIFFUSION_XL = "stabilityai/stable-diffusion-xl-base-1.0"


class Diffuser:
    def __init__(self, model_id: str = StableDiffusionModel.STABLE_DIFFUSION_1_5.value):
        self.model_id = model_id
        self.pipeline = self.initialize_pipeline(model_id)

    @classmethod
    def from_model(cls, model: StableDiffusionModel):
        instance = cls(model.value)
        instance.pipeline = instance.initialize_pipeline(model)
        return instance

    def initialize_pipeline(self, model_id: str):
        return StableDiffusionPipeline.from_pretrained(model_id)

    def generate_image(self, prompt, num_images=1, image_size=512, steps=25):
        images = self.pipeline(prompt, num_images=num_images, image_size=image_size, num_inference_steps=steps)["images"]
        return images


# Beispielnutzung:
def main():
    diffuser = Diffuser()
    prompt = "a photo of an astronaut riding a horse on mars"
    images = diffuser.generate_image(prompt)

    # Speichere das erste generierte Bild
    if images:
        images[0].save("img/output.jpg")


if __name__ == "__main__":
    main()
