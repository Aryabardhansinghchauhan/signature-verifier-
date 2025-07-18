from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import io

app = Flask(__name__)
CORS(app)

# Load TrOCR model from Hugging Face Hub
try:
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    model.eval()
    print("✅ Model and processor loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model or processor: {e}")

# Counter to simulate prediction changes
request_count = 0

@app.route("/upload", methods=["POST"])
def upload_file():
    global request_count
    file = request.files.get("file")

    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        with torch.no_grad():
            generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Simulated prediction
        request_count += 1
        if request_count <= 2:
            label = "Genuine"
        else:
            label = "Forgery"

        return jsonify({
            "label": label,
            "ocr_text": generated_text
        })
    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
