import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import time

from convkan import ConvKAN, LayerNorm2D

# --- Settings ---
DATA_DIR = "/data/pgreshnikov/tiny-imagenet-200"
BATCH_SIZE = 32
EPOCHS = 5
NUM_CLASSES = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Logging ---
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Using device: {DEVICE}")
print(f"Trainable parameters: {count_parameters(model := nn.Sequential())}")

# --- Model ---
model = nn.Sequential(
    ConvKAN(3, 32, padding=1, kernel_size=3, stride=1),
    LayerNorm2D(32),
    ConvKAN(32, 64, padding=1, kernel_size=3, stride=2),
    LayerNorm2D(64),
    ConvKAN(64, NUM_CLASSES, padding=1, kernel_size=3, stride=2),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
).to(DEVICE)

print(f"Using device: {DEVICE}")
print(f"Trainable parameters: {count_parameters(model)}")

# --- Data transforms ---
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Standard ImageNet mean/std
                         std=[0.229, 0.224, 0.225])
])

# --- Load dataset ---
start = time.time()
print("Creating datasets...")
train_dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR, "train"), transform=transform)
val_dataset   = datasets.ImageFolder(root=os.path.join(DATA_DIR, "val"), transform=transform)
test_dataset  = datasets.ImageFolder(root=os.path.join(DATA_DIR, "test"), transform=transform)
print(f"Datasets loaded in {time.time() - start:.2f} seconds")

start = time.time()
print("Creating dataloaders...")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
print(f"Dataloaders created in {time.time() - start:.2f} seconds")

# --- Loss & Optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# --- Training Loop ---
for epoch in range(EPOCHS):
    model.train()
    pbar = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{EPOCHS}] Training")
    for x, y in pbar:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    # --- Validation ---
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_hat = model(x)
            _, predicted = torch.max(y_hat, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    val_acc = 100 * correct / total
    print(f"[Epoch {epoch+1}] Validation Accuracy: {val_acc:.2f}%")

# --- Final Test ---
model.eval()
correct = 0
total = 0
with torch.no_grad():
    pbar = tqdm(test_loader, desc="Testing")
    for x, y in pbar:
        x, y = x.to(DEVICE), y.to(DEVICE)
        y_hat = model(x)
        _, predicted = torch.max(y_hat, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
        pbar.set_postfix(acc=f"{100 * correct / total:.2f}%")

print(f"Final Test Accuracy: {100 * correct / total:.2f}%")
