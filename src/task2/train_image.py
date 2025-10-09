import argparse
import torch
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to config JSON file', default=None)
    parser.add_argument('--data_dir', default='data/animals10_demo')
    parser.add_argument('--output_dir', default='models/image')
    args = parser.parse_args()

    # --- Если есть config.json, читаем его ---
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            cfg = json.load(f)
        args.data_dir = cfg.get("data_dir", args.data_dir)
        args.output_dir = cfg.get("output_dir", args.output_dir)

    # --- Проверяем и создаём папки ---
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Загружаем данные ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    ds = ImageFolder(args.data_dir, transform=transform)
    if len(ds) == 0:
        print('❌ No images found in', args.data_dir)
        return

    loader = DataLoader(ds, batch_size=8, shuffle=True)

    # --- Настройка модели ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(ds.classes))
    model = model.to(device)

    # --- Обучение ---
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    for epoch in range(1):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()

    # --- Сохраняем модель ---
    model_path = os.path.join(args.output_dir, 'resnet18.pth')
    torch.save(model.state_dict(), model_path)
    print(f'✅ Model saved to: {model_path}')


if __name__ == '__main__':
    main()

