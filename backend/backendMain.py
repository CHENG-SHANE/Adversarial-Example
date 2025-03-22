# === Python 標準函式庫 ===
import os
import uuid

# === 第三方函式庫 ===
from flask import Flask, request, jsonify, send_from_directory, url_for
from flask_cors import CORS
import requests

from facenet_pytorch import InceptionResnetV1

from PIL import Image
from matplotlib import pyplot as plt
import matplotlib

# === PyTorch 與 TorchVision ===
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

# ===自訂函式===
from AdversarialExamplesFuntion import GenerateAdversarialExample

# 使用非互動模式的後端
matplotlib.use('Agg')

# 創建 Flask Web
app = Flask(__name__)
CORS(app)

# 資料夾設置
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
STATIC_FOLDER = 'static/charts'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    """檢查圖片格式"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 載入預訓練模型
weights = ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
model.eval() 


preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加載 ImageNet 類別對應表
def load_imagenet_classes():
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

imagenet_classes = load_imagenet_classes()

# 分類器 (Top-5 )
def classify_image_top5(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0)
    output = model(input_tensor)
    probabilities = F.softmax(output, dim=1)
    top5_prob, top5_idx = torch.topk(probabilities, 5, dim=1)
    top5_prob = top5_prob.squeeze().tolist()
    top5_idx = top5_idx.squeeze().tolist()
    top5_classes = [imagenet_classes[idx] for idx in top5_idx]
    return list(zip(top5_classes, top5_prob))

# 圖表
def visualize_predictions(image_path, output_name):
    result_top5 = classify_image_top5(image_path)

    plt.figure(figsize=(8, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(Image.open(image_path))
    plt.axis('off')
    plt.title("Classifier Input", fontsize=16)

    classes, probs = zip(*result_top5)
    probs = [p * 100 for p in probs]  # 轉換為百分比

    plt.subplot(1, 2, 2)
    bars = plt.bar(classes, probs, color="blue")
    plt.ylim(0, 100)
    plt.title("Classifier Output (Top-5)", fontsize=16)
    plt.ylabel("Confidence (%)", fontsize=12)
    plt.xticks(rotation=45, fontsize=10)

    # 在每個長條上標記信心數值
    for bar, prob in zip(bars, probs):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 2, 
                 f"{prob:.2f}%", ha='center', va='bottom', fontsize=10)

    # 保存輸出圖表
    output_path = os.path.join(STATIC_FOLDER, output_name)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return url_for('static', filename=f'charts/{output_name}', _external=True)


# 上傳
@app.route('/upload/', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({"error": "No files part in the request"}), 400
    strength = request.form.get('strength', type=float)
    print("",strength)
    if strength is None:
        return jsonify({"error": "Missing or invalid strength parameter"}), 400

    files = request.files.getlist('files')
    processed_files = []
    classification_results = []

    for file in files:
        if file and allowed_file(file.filename):
            try:
                original_filename = file.filename
                unique_filename = f"{uuid.uuid4().hex}_{original_filename}"
                save_path = os.path.join(UPLOAD_FOLDER, unique_filename)
                file.save(save_path)

                # 原始圖片分類和可視化
                original_chart = visualize_predictions(save_path, f"{uuid.uuid4().hex}_original_chart.png")

                # 處理圖片
                processed_path = GenerateAdversarialExample(save_path, model, preprocess, PROCESSED_FOLDER, strength)
                if processed_path:
                    processed_files.append(os.path.basename(processed_path))
                    processed_chart = visualize_predictions(processed_path, f"{uuid.uuid4().hex}_processed_chart.png")

                    classification_results.append({
                        "original": {"chart": original_chart},
                        "processed": {"chart": processed_chart}
                    })

                    if os.path.exists(save_path):
                        os.remove(save_path)
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        else:
            return jsonify({"error": f"File {file.filename} is not allowed"}), 400

    return jsonify({"processed_files": processed_files, "classification_results": classification_results})

# 下載
@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    try:
        return send_from_directory(PROCESSED_FOLDER, filename, as_attachment=True, mimetype='image/png')
    except FileNotFoundError:
        return jsonify({"error": f"File {filename} not found"}), 404


if __name__ == '__main__':
    app.run(debug=True)
