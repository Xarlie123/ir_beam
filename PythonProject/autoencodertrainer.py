import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 📌 Dataset personalizado en PyTorch
class ImageDataset(Dataset):
    def __init__(self, dataframe, img_size=64):
        self.img_size = img_size
        self.clean_images = np.array([np.array(img, dtype=np.float32).reshape(img_size, img_size) for img in dataframe["original"]])
        self.noisy_images = np.array([np.array(img, dtype=np.float32).reshape(img_size, img_size) for img in dataframe["accumulated"]])

        # Convertir a tensores y aplanar
        self.clean_images = torch.tensor(self.clean_images, dtype=torch.float32).view(-1, img_size * img_size)
        self.noisy_images = torch.tensor(self.noisy_images, dtype=torch.float32).view(-1, img_size * img_size)

    def __len__(self):
        return len(self.clean_images)

    def __getitem__(self, idx):
        return self.noisy_images[idx], self.clean_images[idx]

# 📌 Modelo Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, img_size=64):
        super(Autoencoder, self).__init__()
        self.img_size = img_size
        self.input_dim = img_size * img_size

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, self.input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# 📌 Clase para gestionar entrenamiento y evaluación del Autoencoder
class AutoencoderTrainer:
    def __init__(self, dataframe, img_size=64, batch_size=16, learning_rate=0.001, weight_decay=1e-5):
        self.img_size = img_size
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Cargar dataset
        self.dataset = ImageDataset(dataframe, img_size=img_size)
        train_size = int(0.7 * len(self.dataset))
        val_size = int(0.15 * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, [train_size, val_size, test_size])

        # DataLoaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

        # Modelo, criterio y optimizador
        self.model = Autoencoder(img_size=img_size).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def train(self, num_epochs=100):
        """
        Entrena el autoencoder.
        """
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0

            for noisy_imgs, clean_imgs in self.train_loader:
                noisy_imgs, clean_imgs = noisy_imgs.to(self.device), clean_imgs.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(noisy_imgs)
                loss = self.criterion(outputs, clean_imgs)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            train_loss /= len(self.train_loader)

            # Validación
            val_loss = self.validate()
            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    def validate(self):
        """
        Valida el modelo en el conjunto de validación.
        """
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy_imgs, clean_imgs in self.val_loader:
                noisy_imgs, clean_imgs = noisy_imgs.to(self.device), clean_imgs.to(self.device)
                outputs = self.model(noisy_imgs)
                loss = self.criterion(outputs, clean_imgs)
                val_loss += loss.item()

        return val_loss / len(self.val_loader)

    def save_model(self, path="autoencoder_denoising.pth"):
        """
        Guarda el modelo entrenado.
        """
        torch.save(self.model.state_dict(), path)

    def load_model(self, path="autoencoder_denoising.pth"):
        """
        Carga un modelo previamente entrenado.
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def test_and_plot_results(self):
        """
        Evalúa el modelo en el conjunto de test y grafica los resultados.
        """
        self.model.eval()
        fig, axes = plt.subplots(3, 3, figsize=(9, 9))

        with torch.no_grad():
            for i, (noisy_imgs, clean_imgs) in enumerate(self.test_loader):
                if i >= 3:
                    break

                noisy_imgs, clean_imgs = noisy_imgs.to(self.device), clean_imgs.to(self.device)
                outputs = self.model(noisy_imgs).cpu().numpy().reshape(-1, self.img_size, self.img_size)
                noisy_imgs = noisy_imgs.cpu().numpy().reshape(-1, self.img_size, self.img_size)
                clean_imgs = clean_imgs.cpu().numpy().reshape(-1, self.img_size, self.img_size)

                for j in range(3):
                    axes[j, 0].imshow(noisy_imgs[j], cmap="gray")
                    axes[j, 0].set_title("Imagen Ruidosa")
                    axes[j, 0].axis("off")

                    axes[j, 1].imshow(outputs[j], cmap="gray")
                    axes[j, 1].set_title("Imagen Restaurada")
                    axes[j, 1].axis("off")

                    axes[j, 2].imshow(clean_imgs[j], cmap="gray")
                    axes[j, 2].set_title("Imagen Original")
                    axes[j, 2].axis("off")

        plt.tight_layout()
        plt.show()
