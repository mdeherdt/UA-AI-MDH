# CNN Image Classification 

In this exercise you will:

1. Implement your own Convolutional Neural Network (CNN)
2. Train it on the CIFAR-10 dataset  
3. Evaluate its accuracy  
4. Save the trained model  
5. Load your model and test it on your own images

The project consists of **three files**:

```
main.py                     # Training & evaluation pipeline
model.py                    # You implement the CNN here
predict_with_own_images.py  # Provided: test your model on real images
```

---

## Step 1 Implement `MyCNN` in `model.py`
You can look in the nn_layers.md to see the functions you can use.

Build a small CNN that works on CIFAR-10 (3×32×32 images).

Your model could contain:

- 2–4 convolution layers
- ReLU activations
- One or more MaxPool layers
- A flatten step (`x.view(x.size(0), -1)`)
- One or more fully-connected layers
- Final output layer with **10** units

---

## Step 2 Complete the training loop in `main.py`

The data is already loaded and ready to use. Implement the training loop.
The loss function is defined in criterion.

---

## Step 3 Evaluate your model

Fill in the “testing” section of `main.py`.

The testing loop should look close to the training loop without the training.

Compute accuracy with:

```
correct predictions
total predictions
accuracy = correct / total
```

---

## Step 4 — Save the trained model

Already done for you:

```python
torch.save({
    "model_state_dict": model.state_dict(),
    "classes": classes
}, "cifar10_cnn_student.pth")
```

---

## Step 5 — Test on your own images

You can put any image inside the data folder to test your model. 

The model can recognize :
- plane
- car
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

```
predict_with_own_images.py
```

Put images inside:

```
/data
```

Run:

```
python predict_with_own_images.py
```
