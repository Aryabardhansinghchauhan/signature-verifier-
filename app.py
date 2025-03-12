from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from tqdm import tqdm

# Load the saved model and processor
model_path = r"C:\Users\aryac\Downloads\TrOCR-Signature-Forgery-master\TrOCR-Signature-Forgery-master"
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained(model_path)

# Move model to device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize Flask App
app = Flask(__name__)
CORS(app)  # Enable CORS for all origins

class SignatureDataset(Dataset):
    def __init__(self, images, processor):
        self.images = images
        self.processor = processor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].resize((384, 384))
        image = np.array(image) / 255.0  # Normalize
        if image.shape[-1] != 3:  # Ensure RGB format
            image = np.stack([image] * 3, axis=-1)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # HWC → CHW
        return {"pixel_values": image}

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        try:
            image = Image.open(file).convert("RGB")
            dataset = SignatureDataset([image], processor)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

            model.eval()
            with torch.no_grad():
                for batch in tqdm(dataloader):
                    pixel_values = batch["pixel_values"].to(device)
                    outputs = model.generate(pixel_values)
                    predicted_labels = processor.batch_decode(outputs, skip_special_tokens=True)[0]

            return jsonify({"label": predicted_labels}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Invalid file format"}), 400

if __name__ == '__main__':
    app.run(debug=True)
