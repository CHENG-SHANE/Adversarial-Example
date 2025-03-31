from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import uuid
from PIL import Image, ImageOps
import torch
import numpy as np
from torchvision import transforms
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
PROCESSED_FOLDER = os.path.join(BASE_DIR, 'processed')
STATIC_FOLDER = os.path.join(BASE_DIR, 'static/charts')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
yolo_model = YOLO("D:/Adversarial-Example/model/yoloV8/runs/detect/train6/weights/best.pt")

face_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# FaceNet embedding
def get_embedding(img):
    img_tensor = face_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = facenet(img_tensor)
    return embedding.cpu().numpy()

# YOLO face detection and crop
def detect_and_crop_face(image_path):
    results = yolo_model(image_path, conf=0.3)[0]
    img = Image.open(image_path).convert("RGB")
    faces = []
    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        if cls_id == 0:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            face = img.crop((x1, y1, x2, y2))
            faces.append(((x1, y1, x2, y2), face))
    return img, faces

# Simulate adversarial attack on face region
# def adversarial_attack(face_img, epsilon=0.05):
#     face_tensor = face_transform(face_img).unsqueeze(0).to(device)
#     perturbation = torch.sign(torch.randn_like(face_tensor)) * epsilon
#     adv_face_tensor = face_tensor + perturbation
#     adv_face_tensor = torch.clamp(adv_face_tensor, -1, 1)
#     adv_face_img = transforms.ToPILImage()(adv_face_tensor.squeeze().cpu() * 0.5 + 0.5)
#     return adv_face_img

# Temporary color inversion for debugging
def adversarial_attack(face_img, epsilon=0.05):
    return ImageOps.invert(face_img)

@app.route('/upload/', methods=['POST'])
def upload_files():
    try:
        if 'files' not in request.files:
            return jsonify({"error": "No files part"}), 400

        files = request.files.getlist('files')
        strength = request.form.get('strength', type=float) or 0.05

        processed_files = []
        confidences = []

        for file in files:
            if file and allowed_file(file.filename):
                unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
                save_path = os.path.join(UPLOAD_FOLDER, unique_filename)
                file.save(save_path)

                original_img, faces = detect_and_crop_face(save_path)
                adv_img = original_img.copy()

                for (x1, y1, x2, y2), face in faces:
                    adv_face = adversarial_attack(face, epsilon=strength)
                    adv_face = adv_face.resize((x2 - x1, y2 - y1))
                    adv_img.paste(adv_face, (x1, y1))

                processed_filename = f"adv_{unique_filename}"
                processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)
                adv_img.save(processed_path)

                processed_files.append(processed_filename)

                orig_emb = get_embedding(original_img)
                adv_emb = get_embedding(adv_img)
                cosine_similarity = np.dot(orig_emb, adv_emb.T) / (np.linalg.norm(orig_emb) * np.linalg.norm(adv_emb))
                confidences.append(float(cosine_similarity.squeeze()))

        return jsonify({"processed_files": processed_files, "confidences": confidences})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    folder = PROCESSED_FOLDER if filename.startswith('adv_') else UPLOAD_FOLDER
    return send_from_directory(folder, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
