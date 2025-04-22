import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Fungsi untuk memuat gambar dari folder
def load_image(path, img_size=(128, 128), mode='RGB', scale='tanh'):
    img = Image.open(path).convert(mode)
    img = img.resize(img_size)
    img_array = np.array(img).astype('float32')

    if scale == 'tanh':
        img_array = (img_array / 127.5) - 1.0
    else:
        img_array /= 255.0

    return img_array

# Memuat semua gambar dari folder
def load_images_from_folder(folder, img_size=(128, 128), mode='RGB', scale='tanh', max_images=20):
    images = []
    count = 0
    for filename in os.listdir(folder):
        if filename.endswith(('.jpg', '.png')) and count < max_images:
            img_path = os.path.join(folder, filename)
            img = load_image(img_path, img_size, mode, scale)
            images.append(img)
            count += 1
    return np.array(images)

testA_folder = 'D:/sketch2photo/testA'
testB_folder = 'D:/sketch2photo/testB'

sketch_images = load_images_from_folder(testA_folder, img_size=(128, 128), mode='L', scale='tanh', max_images=20)
color_images = load_images_from_folder(testB_folder, img_size=(128, 128), mode='RGB', scale='tanh', max_images=20)

# Convert grayscale ke RGB (3 channel)
sketch_images = np.repeat(sketch_images[:, :, :, np.newaxis], 3, axis=-1)

# Convert ke tensor
X_tensors = torch.tensor(sketch_images).permute(0, 3, 1, 2).float()  
y_tensors = torch.tensor(color_images).permute(0, 3, 1, 2).float()  

# Define Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Model, loss, optimizer
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(640):
    model.train()
    optimizer.zero_grad()
    output = model(X_tensors)
    loss = criterion(output, y_tensors)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Visualize results
def visualize_results(input_images, output_images, target_images, n=5):
    input_images = (input_images.squeeze().permute(0, 2, 3, 1).numpy() + 1) / 2
    output_images = (output_images.squeeze().permute(0, 2, 3, 1).detach().numpy() + 1) / 2
    target_images = (target_images.squeeze().permute(0, 2, 3, 1).numpy() + 1) / 2

    fig, axs = plt.subplots(n, 3, figsize=(12, 4 * n))
    for i in range(n):
        axs[i, 0].imshow(input_images[i])
        axs[i, 0].set_title('Input (Sketch)')
        axs[i, 0].axis('off')

        axs[i, 1].imshow(output_images[i])
        axs[i, 1].set_title('Output (Generated)')
        axs[i, 1].axis('off')

        axs[i, 2].imshow(target_images[i])
        axs[i, 2].set_title('Target (Color)')
        axs[i, 2].axis('off')

    plt.tight_layout()
    plt.show()

# Show results
model.eval()
with torch.no_grad():
    preds = model(X_tensors)
visualize_results(X_tensors, preds, y_tensors, n=10) 