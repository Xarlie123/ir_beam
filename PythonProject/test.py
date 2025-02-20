from datagenirbeam import DataGenIRBeam
from patterngenerator import PatternGenerator
from maskapplier import MaskApplier
from dataframemanager import TrainingDataframeManager
from autoencodertrainer import AutoencoderTrainer
from CNNregressor import CNNTrainer
from DNNregressor import DNNTrainer

dataset_name = 'ir_profiles_with_params'
img_size = 64  #tamaño de la imagen y de las máscaras en píxeles, son cuadradas. img_size x img_size
number_images_dataset = 1000
pattern_type="combined_sweep"
# pattern_type="scatter"

#Sweep parameters
number_steps = 16 #número de pasos utilizados para hacer las máscaras. Debe ser divisor de img_size

#Scatter parameters
point_density = 0.2
num_patterns = 128

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


##Crea un dataframe con todos los datos que utilizaremos para el entrenamiento

# Crear una instancia del objeto con los datos de entrenamiento
df_manager = TrainingDataframeManager(original_images, parameters, accumulated_images)

# Obtener el DataFrame
df_training = df_manager.get_dataframe()

# Mostrar las primeras filas
print(df_training.head())

# Visualizar algunas imágenes del DataFrame
df_manager.visualize_dataframe_images(img_size=img_size, num_images=3)

## Crear modelo para hacer denoising y entrenarlo

# Crear el objeto de entrenamiento
trainer = AutoencoderTrainer(df_training, img_size=img_size, batch_size=16, learning_rate=0.001)

# Entrenar el modelo
trainer.train(num_epochs=30)

# Guardar el modelo
trainer.save_model("autoencoder_denoising.pth")

# Cargar el modelo (si es necesario en otro momento)
trainer.load_model("autoencoder_denoising.pth")

# Evaluar y graficar resultados en el set de test
trainer.test_and_plot_results()

##Crear modelo para inferir los parámetros y entrenarlo

# Crear el objeto de entrenamiento
trainer = CNNTrainer(df_training, img_size=img_size, batch_size=16, learning_rate=0.001)

# Entrenar el modelo
trainer.train(num_epochs=30)

# Guardar el modelo
trainer.save_model("cnn_regressor.pth")

# Cargar el modelo (si es necesario en otro momento)
trainer.load_model("cnn_regressor.pth")

# Evaluar y graficar resultados en el set de test
trainer.test(num_images = 2)

## Crear modelo con DNN para inferir los parámetros y entrenarlo

# # Crear el objeto de entrenamiento
# trainer = DNNTrainer(df_training, img_size=img_size, batch_size=16, learning_rate=0.001)
#
# # Entrenar el modelo
# trainer.train(num_epochs=25)
#
# # Guardar el modelo
# trainer.save_model("dnn_regressor.pth")
#
# # Cargar el modelo (si es necesario en otro momento)
# trainer.load_model("dnn_regressor.pth")
#
# # Evaluar y graficar resultados en el set de test
# trainer.test(num_images = 2)