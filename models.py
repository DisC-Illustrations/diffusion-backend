from enum import Enum


class DiffusionModel(Enum):
    # STABLE_DIFFUSION_1_5 = "runwayml/stable-diffusion-v1-5"
    STABLE_DIFFUSION_XL = "stabilityai/stable-diffusion-xl-base-1.0"
    STABLE_DIFFUSION_3_5_MEDIUM = "stabilityai/stable-diffusion-3.5-medium"
    FLUX_1_SCHNELL = "black-forest-labs/FLUX.1-schnell"
