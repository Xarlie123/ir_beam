import numpy as np
import matplotlib.pyplot as plt


class PatternGenerator:
    def __init__(self, img_size=64, number_steps=8, point_density=0.1, num_patterns=6):
        self.img_size = img_size
        self.number_steps = number_steps
        self.point_density = point_density
        self.num_patterns = num_patterns

    def generate_combined_sweep_patterns(self):
        if self.img_size % self.number_steps != 0:
            raise ValueError("El número de steps debe ser un divisor exacto del tamaño de la imagen.")

        num_patterns = self.number_steps
        stripe_width = self.img_size // self.number_steps

        H_horizontal = np.zeros((num_patterns, self.img_size, self.img_size))
        H_vertical = np.zeros((num_patterns, self.img_size, self.img_size))
        H_diagonal_45 = np.zeros((num_patterns, self.img_size, self.img_size))
        H_diagonal_neg_45 = np.zeros((num_patterns, self.img_size, self.img_size))

        for i in range(num_patterns):
            H_horizontal[i, i * stripe_width: (i + 1) * stripe_width, :] = 1
            H_vertical[i, :, i * stripe_width: (i + 1) * stripe_width] = 1

            for j in range(self.img_size):
                for k in range(stripe_width * 2):
                    row = j + (i * stripe_width * 2) - self.img_size + k
                    col = j
                    if 0 <= row < self.img_size and 0 <= col < self.img_size:
                        H_diagonal_45[i, row, col] = 1

                    row_neg = (i * stripe_width * 2) + j + k - self.img_size
                    col_neg = self.img_size - 1 - j
                    if 0 <= row_neg < self.img_size and 0 <= col_neg < self.img_size:
                        H_diagonal_neg_45[i, row_neg, col_neg] = 1

        return np.vstack((H_horizontal, H_vertical, H_diagonal_45, H_diagonal_neg_45))

    def generate_scatter_patterns(self):
        if not (0 <= self.point_density <= 1):
            raise ValueError("point_density debe estar en el rango [0, 1]")

        total_pixels = self.img_size * self.img_size
        num_points = int(total_pixels * self.point_density)

        H = np.zeros((self.num_patterns, self.img_size, self.img_size))

        for i in range(self.num_patterns):
            x_coords = np.random.randint(0, self.img_size, num_points)
            y_coords = np.random.randint(0, self.img_size, num_points)
            H[i, x_coords, y_coords] = 1

        return H

    def plot_sweep_patterns(self, dataset_name):
        dataset_path = dataset_name + ".npz"
        intensity_matrices, parameters = self.load_npz_to_dataframe(dataset_path)
        first_image = intensity_matrices[0]

        plt.figure(figsize=(4, 4))
        plt.imshow(first_image, cmap='gray')
        plt.title("Imagen Original")
        plt.colorbar(label="Intensidad")
        plt.show()

        sweep_masks = self.generate_combined_sweep_patterns()
        rows = self.number_steps
        cols = 8
        fig, axes = plt.subplots(rows, cols, figsize=(16, 8))

        group_labels = ["Horizontal", "Vertical", "45°", "-45°"]
        masked_images = [first_image * sweep_masks[i] for i in range(rows * 4)]
        vmin, vmax = np.min(masked_images), np.max(masked_images)

        for i in range(rows):
            for j in range(4):
                mask_idx = i + j * self.number_steps
                mask = sweep_masks[mask_idx]
                masked_image = first_image * mask

                axes[i, j * 2].imshow(mask, cmap='gray', vmin=0, vmax=1)
                axes[i, j * 2].set_title(f"{group_labels[j]} {i + 1}")
                axes[i, j * 2].axis('off')

                axes[i, j * 2 + 1].imshow(masked_image, cmap='gray', vmin=vmin, vmax=vmax)
                axes[i, j * 2 + 1].axis('off')

        plt.tight_layout()
        plt.show()

    def plot_scatter_patterns(self, dataset_name):
        dataset_path = dataset_name + ".npz"
        intensity_matrices, parameters = self.load_npz_to_dataframe(dataset_path)
        first_image = intensity_matrices[0]

        plt.figure(figsize=(4, 4))
        plt.imshow(first_image, cmap='gray')
        plt.title("Imagen Original")
        plt.colorbar(label="Intensidad")
        plt.show()

        scatter_masks = self.generate_scatter_patterns()

        # Ajustar dinámicamente el tamaño de la figura
        fig, axes = plt.subplots(self.num_patterns, 2, figsize=(8, self.num_patterns * 2), constrained_layout=True)

        masked_images = [first_image * scatter_masks[i] for i in range(self.num_patterns)]
        vmin, vmax = np.min(masked_images), np.max(masked_images)

        for i in range(self.num_patterns):
            mask = scatter_masks[i]
            masked_image = first_image * mask

            axes[i, 0].imshow(mask, cmap='gray', vmin=0, vmax=1)
            axes[i, 0].set_title(f"Máscara {i + 1}", fontsize=10)  # Reducir tamaño de fuente
            axes[i, 0].axis('off')

            axes[i, 1].imshow(masked_image, cmap='gray', vmin=vmin, vmax=vmax)
            axes[i, 1].axis('off')

        plt.show()

    def load_npz_to_dataframe(self, dataset_name):
        """
        Carga un dataset .npz y lo convierte en un DataFrame de Pandas.
        """
        data = np.load(dataset_name, allow_pickle=True)

        # Extraer datos
        intensity_matrices = data['intensity_matrices']
        parameters = data['parameters']

        return intensity_matrices, parameters

