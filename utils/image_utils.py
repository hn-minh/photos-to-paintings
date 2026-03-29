import io
import torch
from PIL import Image
from torchvision import transforms

def process_image(image_bytes, device):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor()
    ])
    return transform(img).unsqueeze(0).to(device)

def tensor_to_image(tensor):
    tensor = tensor.cpu().detach().squeeze(0)
    tensor = torch.clamp(tensor, 0, 1)
    transform = transforms.ToPILImage()
    return transform(tensor)