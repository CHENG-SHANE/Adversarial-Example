from ultralytics import YOLO
import torch
import torchvision

print("Torch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Device Count:", torch.cuda.device_count())
print("CUDA Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    model.train(data="data.yaml", epochs=50, batch=16, imgsz=640, device=0)
