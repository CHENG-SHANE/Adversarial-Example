import os
import uuid
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

def GenerateAdversarialExample(image_path, model, preprocess, processed_folder, epsilon):
    try:
        original_image = Image.open(image_path).convert("RGB")
        width, height = original_image.size

        if epsilon is None or epsilon <= 0:
            return None
        max_pixel_value = 1.0
        epsilon = min(epsilon, max_pixel_value * 0.1)

        input_tensor = preprocess(original_image).unsqueeze(0)
        input_tensor.requires_grad = True

        perturbation_limit = epsilon
        max_change_above = input_tensor + perturbation_limit
        max_change_below = input_tensor - perturbation_limit

        output = model(input_tensor)
        _, pred_label = torch.max(output, 1)
        loss = F.cross_entropy(output, pred_label)
        model.zero_grad()
        loss.backward()

        gradient = input_tensor.grad.data
        l2_gradient = gradient / (torch.norm(gradient, p=2) + 1e-10)
        linf_gradient = torch.sign(gradient)

        alpha = 0.9
        combined_gradient = alpha * l2_gradient + (1 - alpha) * linf_gradient

        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(input_tensor.device)
        scaled_epsilon = epsilon / std
        adv_image_tensor = input_tensor + scaled_epsilon * combined_gradient
        adv_image_tensor = torch.clamp(adv_image_tensor, max_change_below, max_change_above)
        adv_image_tensor = torch.clamp(adv_image_tensor, 0, 1)

        beta = max(0.5, 1 - epsilon * 5)
        adv_image_tensor = beta * input_tensor + (1 - beta) * adv_image_tensor

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        adv_image_tensor = adv_image_tensor * std + mean
        adv_image_tensor = torch.clamp(adv_image_tensor, 0, 1)

        adv_image = transforms.ToPILImage()(adv_image_tensor.squeeze())
        adv_image = adv_image.resize((width, height))

        processed_filename = f"processed_{uuid.uuid4().hex}.png"
        processed_path = os.path.join(processed_folder, processed_filename)
        adv_image.save(processed_path)

        return processed_path
    except Exception as e:
        print(e)
        return None
