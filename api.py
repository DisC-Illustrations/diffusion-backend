from flask import Flask, request, jsonify

from diffuser import Diffuser, StableDiffusionModel

app = Flask(__name__)
diffuser = {
    StableDiffusionModel.STABLE_DIFFUSION_XL.value: Diffuser(),
}


@app.route("/generate", methods=["POST"])
def generate_image():
    prompt = request.json["prompt"]
    num_images = request.json.get("num_images", 1)
    image_size = request.json.get("image_size", 1024)
    aspect_ratio = request.json.get("aspect_ratio", 1.0)
    steps = request.json.get("steps", 25)
    model = request.json.get("model", StableDiffusionModel.STABLE_DIFFUSION_XL.value)
    upscale = request.json.get("upscale", 0)

    if aspect_ratio > 1:
        width = image_size
        height = int(image_size / aspect_ratio)
    else:
        width = int(image_size * aspect_ratio)
        height = image_size

    if model not in diffuser:
        try:
            diffuser[model] = Diffuser.from_model(StableDiffusionModel(model))
        except ValueError:
            return jsonify({"error": "Invalid model ID"}), 400

    images = diffuser[model].generate_image(prompt, num_images, width, height, steps, upscale)

    return jsonify({"images": [image.to_base64() for image in images]})
