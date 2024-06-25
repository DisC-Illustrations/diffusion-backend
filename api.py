import base64
import io
import os

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy

from diffuser import Diffuser, StableDiffusionModel

app = Flask(__name__)
CORS(app)

if not os.path.exists(os.path.join(os.path.dirname(__file__), 'db')):
    os.makedirs(os.path.join(os.path.dirname(__file__), 'db'))

basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'db', 'images.sqlite')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


class GeneratedImage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image = db.Column(db.LargeBinary, nullable=False)
    prompt = db.Column(db.String(750), nullable=False)
    negative_prompt = db.Column(db.String(750), nullable=False)
    timestamp = db.Column(db.DateTime, server_default=db.func.now())

    def __repr__(self):
        return f'<GeneratedImage {self.id}>'


with app.app_context():
    db.create_all()


def convert_images(images, image_type="PNG", compression_ratio=0.9):
    converted_images = []
    for i, image in enumerate(images):
        image_bytes = io.BytesIO()

        if image_type == "JPEG":
            image.save(image_bytes, format="JPEG", quality=int(compression_ratio * 100))
        else:
            image.save(image_bytes, format=image_type)

        image_bytes.seek(0)
        image_data = image_bytes.getvalue()
        print(f"Image {i+1} byte size: {len(image_data)}")

        image_base64 = base64.b64encode(image_data).decode('utf-8')
        print(f"Image {i+1} base64 size: {len(image_base64)}")
        print(f"Image {i+1} base64 preview: {image_base64[:80]}...")

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
    color_palette = request.json.get("color_palette", [])

    if not color_palette:
        # Default palette if none provided
        color_palette = [
            {"rgb": [255, 0, 0]},
            {"rgb": [0, 255, 0]},
            {"rgb": [0, 0, 255]},
            {"rgb": [111, 0, 111]},
            {"rgb": [0, 111, 0]},
            {"rgb": [111, 11, 11]},
            {"rgb": [0, 0, 0]},
            {"rgb": [0, 0, 0]}
]

    # Calculate the width and height based on the aspect ratio
    if aspect_ratio > 1:
        width = image_size
        height = int(image_size / aspect_ratio)
    else:
        width = int(image_size * aspect_ratio)
        height = image_size

    # Initialize the diffuser if it doesn't exist
    diffuser = Diffuser.from_model(StableDiffusionModel(model))

    # Generate the images
    images = (diffuser
              .generate_image(prompt, color_palette, negative_prompt, num_images,
                              width, height, steps, upscale))

    converted_images = convert_images(images)

    saved_images = []
    # Save the images to the database
    for image in converted_images:
        decoded_image = base64.b64decode(image)
        print(f"Decoded image size: {len(decoded_image)}")
        new_image = GeneratedImage(image=decoded_image, prompt=prompt, negative_prompt=negative_prompt)
        saved_images.append(new_image)
        db.session.add(new_image)

    db.session.commit()

    # build pairs of image id and base64 image with field names id and image
    result_pairs = [{"id": image.id, "image": image_base64} for image, image_base64 in
                    zip(saved_images, converted_images)]

    # Clear the pipeline and free up memory
    del diffuser

    return jsonify(result_pairs)


@app.route("/images/<int:image_id>", methods=["GET"])
def get_image(image_id):
    image = GeneratedImage.query.get(image_id)
    if image is None:
        return jsonify({"error": "Image not found"}), 404

    print(f"Retrieved image size: {len(image.image)}")
    image_base64 = base64.b64encode(image.image).decode('utf-8')
    print(f"Retrieved image base64 size: {len(image_base64)}")
    print(f"Retrieved image base64 preview: {image_base64[:50]}...")
    return jsonify({"id": image.id, "image": image_base64, "prompt": image.prompt})


@app.route("/images", methods=["GET"])
def get_images():
    images = GeneratedImage.query.all()
    ids = [{"id": image.id} for image in images]
    return jsonify(ids)
