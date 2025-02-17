##
import numpy as np
import matplotlib.pyplot as plt
from datagenirbeam import DataGenIRBeam
from patterngenerator import PatternGenerator
from maskapplier import MaskApplier
from dataframemanager import TrainingDataframeManager
from tqdm import tqdm

dataset_name = 'ir_profiles_with_params'
img_size = 64  #tamaño de la imagen y de las máscaras en píxeles, son cuadradas. img_size x img_size
number_images_dataset = 10
#pattern_type="combined_sweep"
pattern_type="scatter"

#Sweep parameters
number_steps = 8 #número de pasos utilizados para hacer las máscaras. Debe ser divisor de img_size

#Scatter parameters
point_density = 0.2
num_patterns = 32

##Generate Dataset of IR Beam profiler
data_generator = DataGenIRBeam(img_size=img_size, dataset_name=dataset_name)
data_generator.save_dataset(num_images=number_images_dataset)
data_generator.visualize_profiles(num_images=6)


## Pattern generation and plot
pattern_generator = PatternGenerator(img_size=img_size, number_steps=number_steps, point_density=point_density, num_patterns=num_patterns)
pattern_generator.plot_sweep_patterns(dataset_name=dataset_name)
pattern_generator.plot_scatter_patterns(dataset_name=dataset_name)

## Aplica las máscaras a todo el dataset y haz la integral (sumar el valor de cada pixel cubierto y asignarlo a los píxeles de la máscara)
import numpy as np
import matplotlib.pyplot as plt

mask_applier = MaskApplier(pattern_generator=pattern_generator, data_generator=data_generator, pattern_type=pattern_type)  # Cambia a "combined_sweep" si lo deseas

# Ejecutar la función y obtener los resultados
original_images, parameters, accumulated_images = mask_applier.apply_masks_and_accumulate(dataset_name=dataset_name)

# Visualización de resultados
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Mostrar la primera imagen original
axes[0].imshow(original_images[0], cmap='gray')
axes[0].set_title("Imagen Original", fontsize=12)
axes[0].axis('off')

# Mostrar la primera imagen acumulada
axes[1].imshow(accumulated_images[0], cmap='inferno')
axes[1].set_title("Imagen Acumulada", fontsize=12)
axes[1].axis('off')

# Ajustar el diseño y mostrar
plt.tight_layout()
plt.show()

##Crea un dataframe con todos los datos que utilizaremos para el entrenamiento
# Crear una instancia del objeto con los datos de entrenamiento
df_manager = TrainingDataframeManager(original_images, parameters, accumulated_images)

# Obtener el DataFrame
df_training = df_manager.get_dataframe()

# Mostrar las primeras filas
print(df_training.head())

# Visualizar algunas imágenes del DataFrame
df_manager.visualize_dataframe_images(img_size=img_size, num_images=3)

#
# ## Entrenamiento modelo
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader, random_split
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # 📌 Definir Dataset personalizado en PyTorch
# class ImageDataset(Dataset):
#     def __init__(self, dataframe, img_size=64):
#         self.img_size = img_size
#         self.clean_images = np.array([np.array(img, dtype=np.float32).reshape(img_size, img_size) for img in dataframe["original"]])
#         self.noisy_images = np.array([np.array(img, dtype=np.float32).reshape(img_size, img_size) for img in dataframe["accumulated"]])
#
#         # Convertir a tensores y aplanar
#         self.clean_images = torch.tensor(self.clean_images, dtype=torch.float32).view(-1, img_size * img_size)
#         self.noisy_images = torch.tensor(self.noisy_images, dtype=torch.float32).view(-1, img_size * img_size)
#
#     def __len__(self):
#         return len(self.clean_images)
#
#     def __getitem__(self, idx):
#         return self.noisy_images[idx], self.clean_images[idx]
#
# # 📌 Cargar dataset y dividir en train, val, test
# dataset = ImageDataset(df_training, img_size=img_size)
# train_size = int(0.7 * len(dataset))
# val_size = int(0.15 * len(dataset))
# test_size = len(dataset) - train_size - val_size
#
# train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
#
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
#
# # 📌 Mejorar Autoencoder con Batch Normalization y más capas
# class Autoencoder(nn.Module):
#     def __init__(self, img_size=64):
#         super(Autoencoder, self).__init__()
#         self.img_size = img_size
#         self.input_dim = img_size * img_size
#
#         # Encoder
#         self.encoder = nn.Sequential(
#             nn.Linear(self.input_dim, 1024),
#             nn.ReLU(),
#             nn.BatchNorm1d(1024),
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             nn.BatchNorm1d(512),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.BatchNorm1d(256),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.BatchNorm1d(128),
#             nn.Linear(128, 64),
#             nn.ReLU()
#         )
#
#         # Decoder
#         self.decoder = nn.Sequential(
#             nn.Linear(64, 128),
#             nn.ReLU(),
#             nn.BatchNorm1d(128),
#             nn.Linear(128, 256),
#             nn.ReLU(),
#             nn.BatchNorm1d(256),
#             nn.Linear(256, 512),
#             nn.ReLU(),
#             nn.BatchNorm1d(512),
#             nn.Linear(512, 1024),
#             nn.ReLU(),
#             nn.BatchNorm1d(1024),
#             nn.Linear(1024, self.input_dim),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded
#
# # 📌 Configurar modelo, optimizador y loss function
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = Autoencoder(img_size=64).to(device)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
#
# # 📌 Entrenar el Autoencoder
# num_epochs = 100
# for epoch in range(num_epochs):
#     model.train()
#     train_loss = 0.0
#
#     for noisy_imgs, clean_imgs in train_loader:
#         noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
#
#         optimizer.zero_grad()
#         outputs = model(noisy_imgs)
#         loss = criterion(outputs, clean_imgs)
#         loss.backward()
#         optimizer.step()
#
#         train_loss += loss.item()
#
#     train_loss /= len(train_loader)
#
#     # Validación
#     model.eval()
#     val_loss = 0.0
#     with torch.no_grad():
#         for noisy_imgs, clean_imgs in val_loader:
#             noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
#             outputs = model(noisy_imgs)
#             loss = criterion(outputs, clean_imgs)
#             val_loss += loss.item()
#
#     val_loss /= len(val_loader)
#
#     print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
#
# # 📌 Guardar el modelo entrenado
# torch.save(model.state_dict(), "autoencoder_denoising.pth")
#
# # 📌 Evaluar con imágenes de test y graficar resultados
# model.eval()
# fig, axes = plt.subplots(3, 3, figsize=(9, 9))
#
# with torch.no_grad():
#     for i, (noisy_imgs, clean_imgs) in enumerate(test_loader):
#         if i >= 3:
#             break
#
#         noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
#         outputs = model(noisy_imgs).cpu().numpy().reshape(-1, 64, 64)
#         noisy_imgs = noisy_imgs.cpu().numpy().reshape(-1, 64, 64)
#         clean_imgs = clean_imgs.cpu().numpy().reshape(-1, 64, 64)
#
#         for j in range(3):
#             axes[j, 0].imshow(noisy_imgs[j], cmap="gray")
#             axes[j, 0].set_title("Imagen Ruidosa")
#             axes[j, 0].axis("off")
#
#             axes[j, 1].imshow(outputs[j], cmap="gray")
#             axes[j, 1].set_title("Imagen Restaurada")
#             axes[j, 1].axis("off")
#
#             axes[j, 2].imshow(clean_imgs[j], cmap="gray")
#             axes[j, 2].set_title("Imagen Original")
#             axes[j, 2].axis("off")
#
# plt.tight_layout()
# plt.show()