import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import LightPipes as lp


# 📌 Dataset personalizado para inferencia de parámetros
class ParameterDataset(Dataset):
    def __init__(self, dataframe, img_size=64):
        self.img_size = img_size
        self.noisy_images = np.array([
            np.array(img, dtype=np.float32).reshape(1, img_size, img_size)  # Añadir dimensión de canal
            for img in dataframe["accumulated"]
        ])
        self.parameters = np.array([np.array(params, dtype=np.float32) for params in dataframe["parameters"]])

        # Convertir a tensores
        self.noisy_images = torch.tensor(self.noisy_images, dtype=torch.float32)
        self.parameters = torch.tensor(self.parameters, dtype=torch.float32)

    def __len__(self):
        return len(self.noisy_images)

    def __getitem__(self, idx):
        return self.noisy_images[idx], self.parameters[idx]


# 📌 Modelo CNN para inferencia de parámetros
class CNNRegressor(nn.Module):
    def __init__(self):
        super(CNNRegressor, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),  # Ajustar tamaño según img_size
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5)  # 5 parámetros de salida
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Aplanar
        x = self.fc_layers(x)
        return x


# 📌 Clase de entrenamiento
class CNNTrainer:
    def __init__(self, dataframe, img_size=64, batch_size=16, learning_rate=0.001, weight_decay=1e-5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Dataset y DataLoaders
        self.dataset = ParameterDataset(dataframe, img_size=img_size)
        train_size = int(0.7 * len(self.dataset))
        val_size = int(0.15 * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset,
                                                                               [train_size, val_size, test_size])
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

        # Modelo, criterio y optimizador
        self.model = CNNRegressor().to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def train(self, num_epochs=50):
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0

            for noisy_imgs, params in self.train_loader:
                noisy_imgs, params = noisy_imgs.to(self.device), params.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(noisy_imgs)
                loss = self.criterion(outputs, params)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            train_loss /= len(self.train_loader)
            val_loss = self.validate()
            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy_imgs, params in self.val_loader:
                noisy_imgs, params = noisy_imgs.to(self.device), params.to(self.device)
                outputs = self.model(noisy_imgs)
                loss = self.criterion(outputs, params)
                val_loss += loss.item()

        return val_loss / len(self.val_loader)

    def test(self, num_images=1):
        self.model.eval()
        count = 0  # Contador para limitar la cantidad de imágenes graficadas
        with torch.no_grad():
            for noisy_imgs, params in self.test_loader:
                noisy_imgs, params = noisy_imgs.to(self.device), params.to(self.device)
                outputs = self.model(noisy_imgs).cpu().numpy()
                params = params.cpu().numpy()

                for i in range(len(outputs)):
                    if count >= num_images:
                        return  # Detener la ejecución si ya se alcanzó el número deseado
                    self.plot_ellipse(outputs[i], params[i])
                    count += 1

    def plot_ellipse(self, predicted_params, true_params):
        img_size = 64
        size = 15 * lp.mm  # Size of the grid
        wavelength = 410 * lp.nm  # Wavelength of the laser
        N = img_size  # Number of grid points
        R = 3 * lp.mm  # Radius of the beam

        # Asegurar que las variables sean bidimensionales
        predicted_params = np.atleast_2d(predicted_params)
        true_params = np.atleast_2d(true_params)

        # Extract predicted and true parameters
        pred_x, pred_y, pred_sigma_x, pred_sigma_y, pred_angle = predicted_params[0]
        true_x, true_y, true_sigma_x, true_sigma_y, true_angle = true_params[0]

        pred_x *= lp.mm
        pred_y *= lp.mm
        pred_sigma_x *= lp.mm
        pred_sigma_y *= lp.mm

        true_x *= lp.mm
        true_y *= lp.mm
        true_sigma_x *= lp.mm
        true_sigma_y *= lp.mm

        # Create initial Gaussian beam
        F = lp.Begin(size, wavelength, N)
        F = lp.CircAperture(R, 0, 0, F)
        F = lp.GaussBeam(F, w0=0.5 * lp.mm)

        x, y = np.meshgrid(np.linspace(-size / 2, size / 2, N),
                           np.linspace(-size / 2, size / 2, N))

        # Generate Gaussian intensity distributions
        pred_gaussian = np.exp(-((x - pred_x) ** 2 / pred_sigma_x ** 2 + (y - pred_y) ** 2 / pred_sigma_y ** 2))
        true_gaussian = np.exp(-((x - true_x) ** 2 / true_sigma_x ** 2 + (y - true_y) ** 2 / true_sigma_y ** 2))

        F_pred = lp.SubIntensity(pred_gaussian, F)
        I_pred = lp.Intensity(0, F_pred)

        F_true = lp.SubIntensity(true_gaussian, F)
        I_true = lp.Intensity(0, F_true)

        # Plot the generated intensity profiles
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(I_pred, cmap='jet', extent=(-size / 2, size / 2, -size / 2, size / 2))
        axes[0].set_title(f'Predicted Intensity Profile')
        axes[0].set_xlabel('X-axis (mm)')
        axes[0].set_ylabel('Y-axis (mm)')

        axes[1].imshow(I_true, cmap='jet', extent=(-size / 2, size / 2, -size / 2, size / 2))
        axes[1].set_title(f'True Intensity Profile')
        axes[1].set_xlabel('X-axis (mm)')
        axes[1].set_ylabel('Y-axis (mm)')

        plt.colorbar(axes[0].imshow(I_pred, cmap='jet'), ax=axes[0], label='Intensity')
        plt.colorbar(axes[1].imshow(I_true, cmap='jet'), ax=axes[1], label='Intensity')
        plt.show()

    def save_model(self, path="cnn_regressor.pth"):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path="cnn_regressor.pth"):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
