import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
from model import MyCNN

# 1. DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# 2. LOAD CHECKPOINT (MODEL + CLASSES)
checkpoint_path = "cifar10_cnn_student.pth"

checkpoint = torch.load(checkpoint_path, map_location=device)
model = MyCNN().to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

classes = checkpoint.get("classes", (
    'plane', 'car', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
))

print("Loaded model from", checkpoint_path)
print("Classes:", classes)


# 3. TRANSFORM FOR YOUR OWN IMAGES
#    - Resize to 32x32 like CIFAR-10
#    - Convert to tensor
#    - Normalize the same way as during training
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5)),
])


# 4. HELPER: PREDICT SINGLE IMAGE
def predict_image(img_path: str):
    # Open image
    img = Image.open(img_path).convert("RGB")

    # Apply same transforms as training
    tensor = transform(img).unsqueeze(0).to(device)  # shape: (1, 3, 32, 32)

    # Predict
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    predicted_class = classes[pred_idx.item()]
    confidence = conf.item() * 100.0

    print(f"Image: {img_path}")
    print(f"  Predicted: {predicted_class} ({confidence:.1f}% confidence)")
    print()


# 5. RUN ON YOUR OWN IMAGES
#    Put some .jpg or .png in the folder ./data/

# # Option 1: list files manually
# image_paths = [
#     "data/plane.jpg",
#     "data/dog.jpg",
#     # add more here
# ]

# Option 2: automatically take all images in a folder
folder = "data"
image_paths = [
    os.path.join(folder, f)
    for f in os.listdir(folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

for p in image_paths:
    if os.path.exists(p):
        predict_image(p)
    else:
        print(f"File not found: {p}")
