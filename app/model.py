import torch
import torch.nn as nn
from torchvision import models, transforms

def load_model():
    base_model = models.resnet50(pretrained=True)
    model = nn.Sequential(*list(base_model.children())[:-1])
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def get_embedding(image, model):
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model(tensor).squeeze().numpy()
    return embedding.flatten()