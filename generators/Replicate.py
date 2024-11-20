import replicate

def generate_images(prompt: str, width: int = 1024, height: int = 1024, num_inference_steps: int = 15,
                    num_variations: int = 1, replicate_model: str = "FLUX.1-schnell"):
    _input = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_inference_steps": num_inference_steps,
        "num_variations": num_variations,
    }

    replicate.run(
        _input,
        replicate_model
    )

