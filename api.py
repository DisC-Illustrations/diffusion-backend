import io
import base64
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from diffuser import Diffuser, StableDiffusionModel

app = Flask(__name__)
CORS(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db/images.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
diffuser = {
    StableDiffusionModel.STABLE_DIFFUSION_XL.value: Diffuser(),
}


class GeneratedImage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image = db.Column(db.LargeBinary, nullable=False)
    prompt = db.Column(db.String(750), nullable=False)
    negative_prompt = db.Column(db.String(750), nullable=False)
    timestamp = db.Column(db.DateTime, server_default=db.func.now())

    def __repr__(self):
        return f'<GeneratedImage {self.id}>'


def convert_images(images, image_type="PNG", compression_ratio=0.9):
    converted_images = []
    for image in images:
        image_bytes = io.BytesIO()
        if image_type == "JPEG":
            image.save(image_bytes, format="JPEG", quality=int(compression_ratio * 100))
        else:
            image.save(image_bytes, format=image_type)
        image_bytes.seek(0)
        # Encode the BytesIO object to a base64 string
        image_base64 = base64.b64encode(image_bytes.read()).decode('utf-8')
        converted_images.append(image_base64)

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

    saved_images = []
    # Save the images to the database
    for image in converted_images:
        decoded_image = base64.b64decode(image)
        new_image = GeneratedImage(image=decoded_image, prompt=prompt, negative_prompt=negative_prompt)
        saved_images.append(new_image)
        db.session.add(new_image)

    db.session.commit()

    # build pairs of image id and base64 image with field names id and image
    result_pairs = [{"id": image.id, "image": image_base64} for image, image_base64 in
                    zip(saved_images, converted_images)]

    return jsonify(result_pairs)


@app.route("/images/<int:image_id>", methods=["GET"])
def get_image(image_id):
    image = GeneratedImage.query.get(image_id)
    if image is None:
        return jsonify({"error": "Image not found"}), 404

    image_base64 = base64.b64encode(image.image).decode('utf-8')
    return jsonify({"id": image.id, "image": image_base64, "prompt": image.prompt})
