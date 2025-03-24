from ultralytics import YOLO
import os

model_path = "D:/Adversarial Example/model/yoloV8/runs/detect/train6/weights/best.pt" 
image_folder = "funtionTest/v8TestPicture"
output_folder = "funtionTest/v8TestOutput"

model = YOLO(model_path)

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(image_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        image_path = os.path.join(image_folder, filename)
        print(f"inference：{image_path}")
        
        results = model.predict(
            source=image_path,
            conf=0.3,
            save=True,
            save_txt=True,
            project="funtionTest",
            name="v8TestOutput",  
            exist_ok=True
        )

print("finish:", output_folder)
