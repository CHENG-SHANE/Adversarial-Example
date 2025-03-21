import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加載 FaceNet
model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# 數據預處理
transform = transforms.Compose([
    transforms.Resize((160, 160)),  # FaceNet 需要 160x160 的輸入
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def get_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(image_tensor)

    return embedding.cpu().numpy().flatten()

original_embedding = get_embedding("1.jpg")
adv_embedding = get_embedding("1.jpg")

cosine_similarity = np.dot(original_embedding, adv_embedding) / (
    np.linalg.norm(original_embedding) * np.linalg.norm(adv_embedding)
)

print(f": {cosine_similarity:.4f}")
