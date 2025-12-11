# PyTorch Layers & What They Do

Below is a list of most common `torch.nn` modules you can use when building your CNN.  
You **do not need to use all of them**, these are just the essential building blocks.

---

## Convolution Layers

### **`nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)`**
The main workhorse of CNNs.  
A convolution layer scans small filters over the image and extracts features like edges, textures, and shapes.

- `in_channels`: 3 for RGB images, or the number of feature maps from the previous layer  
- `out_channels`: number of filters (features) to learn  
- `kernel_size`: typically 3×3  
- `stride`: how far the kernel moves each step  
- `padding`: adds zeros around the image to preserve size

---

## Activation Functions

### **`nn.ReLU()`**
The most common nonlinear activation. Replaces negative values with 0.

Other activations you *can* use:
- `nn.LeakyReLU()`
- `nn.ELU()`
- `nn.Sigmoid()`
- `nn.Tanh()`

---

## Pooling Layers

### **`nn.MaxPool2d(kernel_size, stride)`**
Reduces spatial size by taking the maximum over each region.

Other pooling:
- `nn.AvgPool2d()`
- `nn.AdaptiveAvgPool2d()` (great for global average pooling)

---

## Normalization

### **`nn.BatchNorm2d(num_features)`**
Normalizes activations to speed up and stabilize training.

---

## Regularization

### **`nn.Dropout(p)`**
Randomly drops neurons with probability `p` to reduce overfitting.

---

## Reshaping Layers

### `nn.Flatten()`  
### **`tensor.view()` / `tensor.reshape()`**

Used to convert conv features `(B, C, H, W)` → `(B, C⋅H⋅W)` for Linear layers.

---

## Fully-Connected Layers

### **`nn.Linear(in_features, out_features)`**
Standard dense layer.  
Used for classification into 10 CIFAR-10 classes.

---

## Loss Functions

### **`nn.CrossEntropyLoss()`**
Use for multi-class classification.

---

## 🛠 Optimizers

- `optim.Adam(...)`  
- `optim.SGD(...)`

---

## Summary Table

| Purpose | Layer | What it does |
|--------|-------|---------------|
| Feature extraction | `nn.Conv2d` | Learns filters for edges and shapes |
| Nonlinearity | `nn.ReLU` | Enables complex patterns |
| Downsampling | `nn.MaxPool2d` | Reduces spatial size |
| Stabilization | `nn.BatchNorm2d` | Normalizes activations |
| Prevent overfitting | `nn.Dropout` | Randomly hides neurons |
| Reshaping | `view` / `Flatten` | Turns features into vectors |
| Classification | `nn.Linear` | Produces class logits |
| Loss | `nn.CrossEntropyLoss` | Computes training error |
| Optimizer | `optim.Adam` | Updates weights |
