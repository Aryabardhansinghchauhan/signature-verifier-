from flask import Flask, request, jsonify
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from tqdm import tqdm

# Load the saved model and processor
model_path = r"C:\Users\aryac\Downloads\TrOCR-Signature-Forgery-master\TrOCR-Signature-Forgery-master"
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained(model_path)

# Move model to device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

app = Flask(__name__)

class SignatureDataset(Dataset):
    def __init__(self, images, processor):
        self.images = images
        self.processor = processor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        inputs = self.processor(images=image, return_tensors="pt")  # Use processor directly
        return {"pixel_values": inputs["pixel_values"].squeeze(0)}  # Remove batch dim

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

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
                    predicted_labels = processor.batch_decode(outputs, skip_special_tokens=True)

            # Ensure result is not empty
            label = predicted_labels[0] if predicted_labels else "No prediction"

            return jsonify({"label": label}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Invalid file format"}), 400

if __name__ == '__main__':
    app.run(debug=True)
