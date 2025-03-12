from transformers import VisionEncoderDecoderModel  # ✅ Import the model

model_path = r"C:\Users\aryac\Downloads\TrOCR-Signature-Forgery-master\TrOCR-Signature-Forgery-master"

model = VisionEncoderDecoderModel.from_pretrained(model_path)  # ✅ No more NameError
