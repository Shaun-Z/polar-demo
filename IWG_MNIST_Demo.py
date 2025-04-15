
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

# Load MNIST with light transform
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
subset_indices = list(range(200))  # smaller subset for demo
mnist_subset = Subset(mnist, subset_indices)
data_loader = DataLoader(mnist_subset, batch_size=32, shuffle=True)

# Define simple MLP model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Initialize and train model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
model.train()
for batch_x, batch_y in data_loader:
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    output = model(batch_x)
    loss = F.cross_entropy(output, batch_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Define compute_IWG
def compute_IWG(model, x, target_class, pixel_idx, data_loader, num_samples=50):
    model.eval()
    x = x.to(device)
    x_pixel = x[0, pixel_idx].item()
    grads_list = []

    # Gather samples for KDE
    all_x = []
    for batch_x, _ in data_loader:
        all_x.append(batch_x)
    all_x = torch.cat(all_x, dim=0).numpy()
    xi_all = all_x[:, pixel_idx].reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(xi_all)

    count = 0
    for batch_x, _ in data_loader:
        batch_x = batch_x.to(device)
        batch_x.requires_grad = True
        outputs = model(batch_x)
        probs = F.softmax(outputs, dim=1)[:, target_class]
        grads = torch.autograd.grad(probs.sum(), batch_x)[0][:, pixel_idx]
        
        # Move everything to CPU and convert to numpy for calculation
        delta_xi = x_pixel - batch_x[:, pixel_idx].detach().cpu().numpy()
        grads_cpu = grads.detach().cpu().numpy()
        weight = np.exp(kde.score_samples(batch_x[:, pixel_idx].detach().cpu().numpy().reshape(-1, 1)))
        
        # Perform calculation in numpy domain
        iwg_contrib = delta_xi * grads_cpu * weight
        grads_list.extend(iwg_contrib)
        
        count += len(iwg_contrib)
        if count >= num_samples:
            break

    return np.mean(grads_list)

# Use a test sample
x0, y0 = mnist[0]
x0 = x0[None, ...]
target_class = y0

# Compute full attribution map
attributions = np.zeros(784)
for i in range(784):
    attributions[i] = compute_IWG(model, x0, target_class, i, data_loader, num_samples=50)

# Visualize full heatmap
heatmap = attributions.reshape(28, 28)
plt.figure(figsize=(5, 5))
plt.imshow(heatmap, cmap='hot')
plt.colorbar()
plt.title("Full IWG Heatmap")
plt.tight_layout()
plt.savefig("iwg_heatmap_full.png")
print("Saved to iwg_heatmap_full.png")
