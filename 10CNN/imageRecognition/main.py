import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from model import MyCNN


# 1. DEVICE SETUP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# 2. LOAD CIFAR-10 DATASET
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(
    root='./train_data',
    train=True,
    download=True,
    transform=transform
)
testset = torchvision.datasets.CIFAR10(
    root='./train_data',
    train=False,
    download=True,
    transform=transform
)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


# 3. INITIALIZE MODEL
model = MyCNN().to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 4. TRAINING LOOP  (TODO)
EPOCHS = 5
print("\nTraining...\n")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in trainloader:
        images = images.to(device)
        labels = labels.to(device)
        # TODO: implement full training loop
        # You can keep track of the loss to print it with running_loss


    print(f"Epoch {epoch+1}/{EPOCHS} | Loss = {running_loss:.4f}")


# 5. TESTING LOOP (TODO)
correct = 0
total = 0

model.eval()

with torch.no_grad():
    for images, labels in testloader:
        pass
        # TODO: implement evaluation loop


print(f"Accuracy: {100 * correct / total:.2f}%")


# 6. SAVE MODEL (already implemented)
save_path = "cifar10_cnn_student.pth"

checkpoint = {
    "model_state_dict": model.state_dict(),
    "classes": classes,
}
torch.save(checkpoint, save_path)

print(f"Model saved to {save_path}")
