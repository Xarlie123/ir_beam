import numpy as np
import matplotlib.pyplot as plt

class MaskApplier:
    def __init__(self, pattern_generator, data_generator, pattern_type="scatter"):
        """
        Inicializa el objeto MaskApplier con el tipo de patrón a generar.

        Parámetros:
        - pattern_type (str): Tipo de patrón de máscara. Puede ser "scatter" o "combined_sweep".
        """
        self.pattern_type = pattern_type
        self.pattern_generator = pattern_generator
        self.data_generator = data_generator


    def generate_masks(self):
        """
        Genera las máscaras según el tipo de patrón seleccionado.

        Retorna:
        - Lista de máscaras generadas.
        """
        if self.pattern_type == "scatter":
            return self.pattern_generator.generate_scatter_patterns()
        elif self.pattern_type == "combined_sweep":
            return self.pattern_generator.generate_combined_sweep_patterns()
        else:
            raise ValueError("Tipo de patrón no reconocido. Usa 'scatter' o 'combined_sweep'.")

    def apply_masks_and_accumulate(self, dataset_name):
        """
        Carga un dataset de imágenes, aplica todas las máscaras a la primera imagen
        y acumula los valores de los píxeles después de cada aplicación.

        Parámetros:
        - dataset_name: Nombre del archivo .npz con los datos.

        Retorna:
        - intensity_matrices: Imágenes originales.
        - parameters: Parámetros del dataset.
        - accumulated_images: Imágenes con máscaras aplicadas y acumuladas.
        """

        # Cargar el dataset
        dataset_path = dataset_name + ".npz"
        intensity_matrices, parameters = self.data_generator.load_npz_to_dataframe(dataset_path)

        # Generar las máscaras
        sweep_masks = self.generate_masks()

        # Inicializar la imagen acumulativa con ceros
        accumulated_images = np.zeros_like(intensity_matrices)

        # Aplicar cada máscara y acumular los valores
        for idx, image in enumerate(intensity_matrices):
            for mask in sweep_masks:
                masked_image = image * mask  # Aplicar la máscara
                sum_masked_pixels = masked_image.sum()  # Sumar todos los píxeles enmascarados
                accumulated_images[idx] += sum_masked_pixels * mask  # Acumular en la imagen correspondiente

        return intensity_matrices, parameters, accumulated_images  # Retorna imágenes originales y acumuladas
