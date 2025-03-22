from ultralytics import YOLO
import os
import cv2
import numpy as np

model_path = "D:/Adversarial-Example/model/yoloV8/runs/detect/train6/weights/best.pt"
image_folder = "D:/Adversarial-Example/funtionTest/corpandpast/cropAndPastPicture"
output_folder = "D:/Adversarial-Example/funtionTest/corpandpast/cropAndPastOutput"
target_class_id = 0

os.makedirs(output_folder, exist_ok=True)

model = YOLO(model_path)

# 處理圖片
for filename in os.listdir(image_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):

        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        results = model(img_path, conf=0.3)[0]

        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            if cls_id == target_class_id:

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                # 裁切人臉區域
                face_bgr = img[y1:y2, x1:x2]

                # 顏色反轉
                inverted_face = 255 - face_bgr

                # 將反轉後的區域貼回原圖
                img[y1:y2, x1:x2] = inverted_face

        # 儲存結果圖片
        output_path = os.path.join(output_folder, filename.rsplit('.', 1)[0] + "_inverted.png")
        cv2.imwrite(output_path, img)
        print(f"finish：{output_path}")
