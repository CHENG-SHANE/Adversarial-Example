import torch
import numpy as np
import matplotlib.pyplot as plt
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from PIL import Image
from matplotlib.patches import ConnectionPatch
import os

# 載入FaceNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# 圖片前處理
transform = transforms.Compose([
    transforms.Resize((160, 160)), 
    transforms.ToTensor(),          
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# 取得圖片的FaceNet嵌入向量
def get_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(image_tensor)
    return embedding.cpu().numpy().flatten()

# 載入並回傳圖片
def load_image(image_path):
    return Image.open(image_path).convert("RGB")

# 圖片配對
pairs = [(i, i+1) for i in range(1, 8, 2)]
num_pairs = len(pairs)

base_dir = r"D:\Adversarial-Example\funtionTest\testFaceNet"
pairs = [(i, i+1) for i in range(1, 8, 2)]
num_pairs = len(pairs)

# 顯示原始圖片與對抗圖片
fig, axes = plt.subplots(nrows=num_pairs, ncols=2, figsize=(10, num_pairs*4))
if num_pairs == 1:
    axes = np.array([axes]) 

plt.subplots_adjust(hspace=0.5, top=0.9)

fig.suptitle("FaceNet Adversarial Example Attack\n", fontsize=14)
fig.text(0.5, 0.92, "The confidence value indicates whether two pictures are of the same person.\n",
         ha="center", fontsize=12)

for i, (orig_idx, adv_idx) in enumerate(pairs):
    orig_path = os.path.join(base_dir, f"{orig_idx}.jpg")
    adv_path = os.path.join(base_dir, f"{adv_idx}.jpg")
    
    emb_orig = get_embedding(orig_path)
    emb_adv = get_embedding(adv_path)
    cosine_similarity = np.dot(emb_orig, emb_adv) / (np.linalg.norm(emb_orig) * np.linalg.norm(emb_adv))
    
    image_orig = load_image(orig_path)
    image_adv = load_image(adv_path)
    
    axes[i, 0].imshow(image_orig)
    axes[i, 0].set_title("before", fontsize=12)
    axes[i, 0].axis("off")
    
    axes[i, 1].imshow(image_adv)
    axes[i, 1].set_title("after", fontsize=12)
    axes[i, 1].axis("off")
    
    con = ConnectionPatch(xyA=(1.6, 0.5), coordsA=axes[i, 0].transAxes,
                          xyB=(-0.6, 0.5), coordsB=axes[i, 1].transAxes,
                          arrowstyle="->", color="black", lw=4)
    fig.add_artist(con)
    
    left_arrow = axes[i, 0].transAxes.transform((1.6, 0.5))
    right_arrow = axes[i, 1].transAxes.transform((-0.6, 0.5))
    midpoint = ((left_arrow[0] + right_arrow[0]) / 2, (left_arrow[1] + right_arrow[1]) / 2)
    midpoint_fig = fig.transFigure.inverted().transform(midpoint)
    
    fig.text(midpoint_fig[0], midpoint_fig[1] + 0.03, f"Confidence: {cosine_similarity:.4f}",
             ha="center", va="bottom", fontsize=12)
    
    print(f"FaceNet similarity ({orig_idx}.jpg vs {adv_idx}.jpg): {cosine_similarity:.4f}")

plt.show()