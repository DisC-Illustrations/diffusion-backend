import io

from flask import Flask, request, jsonify

from diffuser import Diffuser, StableDiffusionModel

app = Flask(__name__)
diffuser = {
    StableDiffusionModel.STABLE_DIFFUSION_XL.value: Diffuser(),
}


def convert_images(images, image_type="PNG", compression_ratio=0.9):
    converted_images = []
    for image in images:
        image_bytes = io.BytesIO()
        if image_type == "JPEG":
            image.save(image_bytes, format="JPEG", quality=compression_ratio * 100)
        else:
            image.save(image_bytes, format=image_type)
        image_bytes.seek(0)
        converted_images.append(image_bytes)

    return converted_images


@app.route("/generate", methods=["POST"])
def generate_image():
    prompt = request.json["prompt"]
    negative_prompt = request.json.get("negative_prompt", "text, watermarks")
    num_images = request.json.get("num_images", 1)
    image_size = request.json.get("image_size", 1024)
    aspect_ratio = request.json.get("aspect_ratio", 1.0)
    steps = request.json.get("steps", 25)
    model = request.json.get("model", StableDiffusionModel.STABLE_DIFFUSION_XL.value)
    upscale = request.json.get("upscale", 1)

    # Calculate the width and height based on the aspect ratio
    if aspect_ratio > 1:
        width = image_size
        height = int(image_size / aspect_ratio)
    else:
        width = int(image_size * aspect_ratio)
        height = image_size

    # Initialize the diffuser if it doesn't exist
    if model not in diffuser:
        try:
            diffuser[model] = Diffuser.from_model(StableDiffusionModel(model))
        except ValueError:
            return jsonify({"error": "Invalid model ID"}), 400

    # Generate the images
    images = (diffuser[model]
              .generate_image(prompt, negative_prompt, num_images,
                              width, height, steps, upscale))

    converted_images = convert_images(images)

    return jsonify({"images": [image for image in converted_images]})
