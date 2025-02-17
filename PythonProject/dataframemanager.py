import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class TrainingDataframeManager:
    def __init__(self, original_images, parameters, accumulated_images):
        """
        Clase para gestionar la creación y visualización del DataFrame de entrenamiento.

        Parámetros:
        - original_images: Lista o array de imágenes originales.
        - parameters: Array NumPy de parámetros asociados a cada imagen.
        - accumulated_images: Lista o array de imágenes acumuladas.
        """
        self.df = self.create_training_dataframe(original_images, parameters, accumulated_images)

    @staticmethod
    def create_training_dataframe(original_images, parameters, accumulated_images):
        """
        Crea un DataFrame con imágenes originales, parámetros y las imágenes acumuladas.

        Parámetros:
        - original_images: Lista o array de imágenes originales.
        - parameters: Array NumPy de parámetros asociados a cada imagen.
        - accumulated_images: Lista o array de imágenes acumuladas.

        Retorna:
        - df: DataFrame con tres columnas ('original', 'parameters', 'accumulated').
        """

        # Convertir imágenes en listas aplanadas (1D)
        original_flattened = [img.flatten().tolist() for img in original_images]
        accumulated_flattened = [img.flatten().tolist() for img in accumulated_images]

        # Convertir parameters a lista estándar
        parameters_list = parameters.tolist() if isinstance(parameters, np.ndarray) else parameters

        # Crear DataFrame
        df = pd.DataFrame({
            "original": original_flattened,  # Listas aplanadas de imágenes originales
            "parameters": parameters_list,  # Parámetros convertidos a lista estándar
            "accumulated": accumulated_flattened  # Listas aplanadas de imágenes acumuladas
        })

        return df

    def visualize_dataframe_images(self, img_size=64, num_images=3):
        """
        Visualiza las primeras imágenes del DataFrame en un plot con 3 columnas:
        - Imagen original (reshape a img_size x img_size)
        - Parámetros (expresados con su significado, incluyendo el ángulo en radianes y grados)
        - Imagen acumulada (reshape a img_size x img_size)

        Parámetros:
        - img_size: Tamaño de la imagen cuadrada (img_size x img_size).
        - num_images: Número de imágenes a visualizar.
        """

        # Asegurar que num_images no sea mayor al tamaño del dataset
        num_images = min(num_images, len(self.df))

        fig, axes = plt.subplots(num_images, 3, figsize=(12, 4 * num_images))

        for i in range(num_images):
            # Extraer y reconstruir la imagen original
            original = np.array(self.df.loc[i, "original"]).reshape((img_size, img_size))
            accumulated = np.array(self.df.loc[i, "accumulated"]).reshape((img_size, img_size))
            parameters = self.df.loc[i, "parameters"]

            # Extraer los valores de los parámetros con significado
            center_x, center_y, sigma_x, sigma_y, angle_rad = parameters
            angle_deg = np.degrees(angle_rad)  # Convertir a grados

            # Texto con la interpretación de los parámetros
            param_text = (
                f"Center X: {center_x:.2f} mm\n"
                f"Center Y: {center_y:.2f} mm\n"
                f"Sigma X: {sigma_x:.2f} mm\n"
                f"Sigma Y: {sigma_y:.2f} mm\n"
                f"Angle: {angle_rad:.2f} rad ({angle_deg:.1f}°)"
            )

            # Plot de la imagen original
            axes[i, 0].imshow(original, cmap="gray")
            axes[i, 0].set_title("Imagen Original")
            axes[i, 0].axis("off")

            # Mostrar los parámetros en texto
            axes[i, 1].text(0.5, 0.5, param_text, fontsize=12, ha="center", va="center")
            axes[i, 1].set_title("Parámetros")
            axes[i, 1].axis("off")

            # Plot de la imagen acumulada
            axes[i, 2].imshow(accumulated, cmap="inferno")
            axes[i, 2].set_title("Imagen Acumulada")
            axes[i, 2].axis("off")

        plt.tight_layout()
        plt.show()

    def get_dataframe(self):
        """
        Retorna el DataFrame almacenado.

        Retorna:
        - DataFrame con imágenes originales, parámetros y acumuladas.
        """
        return self.df
