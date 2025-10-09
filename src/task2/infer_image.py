from torchvision import models, transforms
from PIL import Image
import torch
import os

def predict_image(image_path, model_path=None, classes=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = models.resnet18(pretrained=False)
    if model_path and os.path.exists(model_path):
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
    model.eval()
    transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    img = Image.open(image_path).convert('RGB')
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
    if classes:
        top = probs.argmax()
        return classes[top], float(probs[top])
    return None, None
