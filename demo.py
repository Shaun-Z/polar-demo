import torch
import torchvision
import torchvision.transforms as transforms

# Define transforms: resize to 224 for ResNet, convert to tensor, normalize to ImageNet mean/std
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),            # Resize CIFAR-10 image from 32x32 to 224x224
    transforms.ToTensor(),                   # Convert PIL image to PyTorch tensor ([0,1] range)
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # Normalize with ImageNet statistics
                         std=[0.229, 0.224, 0.225])
])
test_transform = train_transform  # use same transforms for test data

# Download CIFAR-10 training and test sets
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
testset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

# Create data loaders for batching (if needed for training or evaluation)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader  = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# (For feature analysis, we'll also access raw images without transforms)
raw_testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=None)

# %%
import torch.nn as nn
from torchvision import models

# Load a pretrained ResNet-18 model
model = models.resnet18(pretrained=True)
model.eval()  # set to evaluation mode by default (since we will fine-tune or evaluate)

# Modify the final fully connected layer to have 10 output classes (instead of 1000)
num_features = model.fc.in_features   # number of features in original fc layer
model.fc = nn.Linear(num_features, 10)

# (Optional) If using the model as a fixed feature extractor, freeze convolutional layers:
# for param in model.parameters():
#     param.requires_grad = False
# model.fc.requires_grad = True  # only train the final layer

# %%
import numpy as np

def extract_patch_features(image: np.ndarray, patch_size: int = 4) -> np.ndarray:
    """
    Split a 32x32 image into patches of size patch_size x patch_size and 
    return an array of binary feature values (0 or 1) for each patch 
    based on mean intensity thresholding.
    """
    # Ensure image is numpy array of shape (32, 32, 3) in [0,1] range
    H, W, C = image.shape
    assert H == W == 32, "Expected 32x32 image"
    # Convert to grayscale by averaging channels
    gray = image.mean(axis=2)  # shape (32, 32)
    h = w = patch_size
    features = []
    # Loop over patches
    for i in range(0, H, h):
        for j in range(0, W, w):
            patch = gray[i:i+h, j:j+w]           # select patch region
            patch_mean = patch.mean()           # compute mean intensity
            feature = 1 if patch_mean > 0.5 else 0   # binary feature (bright=1 if mean > 0.5)
            features.append(feature)
    return np.array(features, dtype=int)

# Example: extract patches from the first image in the test set
img_pil, label = raw_testset[0]           # get raw PIL image (32x32)
img_arr = np.array(img_pil) / 255.0       # convert to numpy array and scale to [0,1]
patch_feats = extract_patch_features(img_arr, patch_size=4)
print(f"Extracted {len(patch_feats)} patch features: {patch_feats}")

# %%

def polar_transform(bits: np.ndarray) -> np.ndarray:
    """
    Perform in-place polar transformation on a binary array of length N (where N is a power of 2).
    Returns a new numpy array of the same length with transformed bits.
    """
    bits = bits.copy().astype(int)  # work on a copy, ensure integer type (0/1)
    N = bits.shape[0]
    assert (N & (N-1)) == 0, "Length of input must be power of 2 for Polar transform"
    step = 1
    # Iteratively apply XOR combinations in log2(N) stages
    while step < N:
        for i in range(0, N, 2*step):
            for j in range(i, i+step):
                bits[j] = bits[j] ^ bits[j+step]
        step *= 2
    return bits

# Example: apply polar transform to the patch features of the first test image
polar_feats = polar_transform(patch_feats)
print(f"Polar transformed features: {polar_feats}")

# %%

import math

def estimate_feature_capacity(dataset, patch_size: int = 4) -> np.ndarray:
    """
    Estimate capacity (importance) of each polar-transformed feature by entropy.
    Returns an array of entropy values for each feature index.
    """
    # Determine number of features from one image (power of 2, e.g., 64)
    sample_img, _ = dataset[0]
    img_arr = np.array(sample_img) / 255.0
    u = extract_patch_features(img_arr, patch_size=patch_size)
    N = len(u)
    # Counter for feature bit =1 occurrences
    count_ones = np.zeros(N, dtype=int)
    total = 0
    # Loop through dataset images
    for idx in range(len(dataset)):
        img, _ = dataset[idx]
        img = np.array(img) / 255.0
        u = extract_patch_features(img, patch_size=patch_size)
        x = polar_transform(u)
        count_ones += x  # add 1s for each feature position
        total += 1
    # Calculate entropy for each feature
    capacities = []
    for k in range(len(count_ones)):
        p = count_ones[k] / total  # probability that feature k is 1
        if p == 0 or p == 1:
            entropy = 0.0
        else:
            entropy = -(p * math.log2(p) + (1-p) * math.log2(1-p))
        capacities.append(entropy)
    return np.array(capacities)

# Estimate capacities on the test set (this may take a bit of time for the entire set)
feature_capacities = estimate_feature_capacity(raw_testset, patch_size=4)
print("Feature entropy (capacity) scores:", feature_capacities)

# %%
def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Normalize an array of feature scores to the range [0, 1]."""
    min_val = scores.min()
    max_val = scores.max()
    if max_val - min_val < 1e-6:
        # Avoid division by zero if all scores equal
        return np.zeros_like(scores)
    normalized = (scores - min_val) / (max_val - min_val)
    return normalized

normalized_capacities = normalize_scores(feature_capacities)
print("Normalized feature scores:", normalized_capacities)

# %%

import matplotlib.pyplot as plt

# Plot feature importance scores
plt.figure(figsize=(8,4))
plt.bar(range(len(normalized_capacities)), normalized_capacities, color='skyblue')
plt.title("Feature Importance (Polar Channel Capacity per Patch)")
plt.xlabel("Feature Index")
plt.ylabel("Normalized Capacity Score")
plt.tight_layout()
plt.show()

# %%

# Choose a target layer for Grad-CAM – typically the last convolutional layer
target_layer = model.layer4  # layer4 is the final set of conv layers in ResNet-18

# Placeholders to store features and gradients
features = None
grads = None

# Define forward and backward hooks
def forward_hook(module, input, output):
    # Save the feature maps output from target layer
    global features
    features = output.detach()

def backward_hook(module, grad_input, grad_output):
    # Save the gradients flowing back in this layer
    global grads
    grads = grad_output[0].detach()

# Register hooks
forward_handle = target_layer.register_forward_hook(forward_hook)
backward_handle = target_layer.register_backward_hook(backward_hook)

# Get a sample image and label from the test set
sample_img, sample_label = testset[0]        # already transformed (tensor normalized)
sample_img = sample_img.unsqueeze(0)         # add batch dimension
# Forward pass
outputs = model(sample_img)
pred_class = outputs.argmax(dim=1).item()    # predicted class index
# Backward pass for the predicted class score
model.zero_grad()
outputs[0, pred_class].backward()

# Now 'features' holds the feature map of the target layer, 'grads' holds the gradients
# Compute weights: global average of gradients for each feature map channel
weights = grads.mean(dim=(2, 3), keepdim=True)  # shape [batch, channels, 1, 1]
# Weighted combination of feature maps
gradcam_map = (features * weights).sum(dim=1, keepdim=True)  # shape [1, 1, H, W]
gradcam_map = torch.nn.functional.relu(gradcam_map)         # apply ReLU
gradcam_map = gradcam_map.squeeze().cpu().numpy()           # shape [H, W]

# Normalize the heatmap to [0,1]
gradcam_map -= gradcam_map.min()
if gradcam_map.max() > 0:
    gradcam_map /= gradcam_map.max()

print("Grad-CAM heatmap (raw values):")
print(gradcam_map)

# %%

# Hook the first conv layer and first layer block to get their outputs
conv1_features = None
layer1_features = None
def conv1_hook(module, inp, out):
    global conv1_features
    conv1_features = out.detach()
def layer1_hook(module, inp, out):
    global layer1_features
    layer1_features = out.detach()

# Register hooks on model.conv1 and model.layer1
h1 = model.conv1.register_forward_hook(conv1_hook)
h2 = model.layer1.register_forward_hook(layer1_hook)

# Forward pass the sample image through the model (same sample_img from before)
_ = model(sample_img)

# Remove hooks
h1.remove()
h2.remove()

print("Conv1 feature maps output shape:", conv1_features.shape)
print("Layer1 output feature maps shape:", layer1_features.shape)

# %%

# Visualize a few conv1 feature maps (if we had plotting capabilities)
fig, axes = plt.subplots(2, 5, figsize=(10,4))
for i, ax in enumerate(axes.flat):
    if i < conv1_features.shape[1]:
        # Take the feature map of channel i
        fm = conv1_features[0, i].cpu().numpy()
        # Normalize for display
        fm -= fm.min()
        fm /= (fm.max() + 1e-5)
        ax.imshow(fm, cmap='gray')
        ax.axis('off')
plt.suptitle("Sample feature maps from ResNet-18 conv1")
plt.show()