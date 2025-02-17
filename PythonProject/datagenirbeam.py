import numpy as np
import matplotlib.pyplot as plt
import LightPipes as lp
from tqdm import tqdm


class DataGenIRBeam:
    def __init__(self, img_size=64, dataset_name='ir_profiles_with_params'):
        self.img_size = img_size
        self.dataset_name = dataset_name
        self.size = 15 * lp.mm  # Size of the grid
        self.wavelength = 410 * lp.nm  # Wavelength of the laser
        self.N = img_size  # Number of grid points
        self.R = 3 * lp.mm  # Radius of the beam

    def generate_intensity_profiles(self, num_images):
        # Initialize arrays to store intensity matrices and parameters
        intensity_matrices = np.zeros((num_images, self.N, self.N))
        parameters = np.zeros((num_images, 5))

        # Create an initial Gaussian beam with a larger radius
        F = lp.Begin(self.size, self.wavelength, self.N)
        F = lp.CircAperture(self.R, 0, 0, F)  # Centered aperture
        F = lp.GaussBeam(F, w0=0.5 * lp.mm)

        # Loop to generate and save different intensity profiles
        for i in tqdm(range(num_images), desc="Generating intensity profiles"):
            # Randomize parameters
            center_x = np.random.uniform(-3, 3) * lp.mm
            center_y = np.random.uniform(-3, 3) * lp.mm
            sigma_x = np.random.uniform(1, 3) * lp.mm
            sigma_y = np.random.uniform(1, 3) * lp.mm
            angle_ = np.random.uniform(-np.pi / 2, np.pi / 2)  # Angle of rotation

            # Store parameters
            parameters[i] = [center_x / lp.mm, -center_y / lp.mm, sigma_x / lp.mm, sigma_y / lp.mm, angle_]

            # Apply a non-symmetric intensity modulation to the Gaussian beam
            x, y = np.meshgrid(np.linspace(-self.size / 2, self.size / 2, self.N),
                               np.linspace(-self.size / 2, self.size / 2, self.N))

            # Rotate coordinates around the center
            x_rot = (x - center_x) * np.cos(angle_) - (y - center_y) * np.sin(angle_) + center_x
            y_rot = (x - center_x) * np.sin(angle_) + (y - center_y) * np.cos(angle_) + center_y

            # Gaussian intensity distribution with elliptical shape
            gaussian = np.exp(-((x_rot - center_x) ** 2 / sigma_x ** 2 + (y_rot - center_y) ** 2 / sigma_y ** 2))
            gaussian = np.clip(gaussian, 0, 1)  # Clip values to ensure they are within the valid range

            F_mod = lp.SubIntensity(gaussian, F)
            I = lp.Intensity(0, F_mod)  # Convert the result to intensity

            # Store the intensity matrix
            intensity_matrices[i] = I

        return intensity_matrices, parameters

    def save_dataset(self, num_images):
        intensity_matrices, parameters = self.generate_intensity_profiles(num_images)
        np.savez(self.dataset_name + '.npz', intensity_matrices=intensity_matrices, parameters=parameters)
        print(f"Dataset saved as {self.dataset_name}.npz")

    def visualize_profiles(self, num_images=6):
        intensity_matrices, parameters = self.generate_intensity_profiles(num_images)
        plt.figure(figsize=(15, 10))
        for i in range(min(6, num_images)):
            params = parameters[i]
            plt.subplot(2, 3, i + 1)
            plt.imshow(intensity_matrices[i],
                       extent=[-self.size / 2 / lp.mm, self.size / 2 / lp.mm, -self.size / 2 / lp.mm,
                               self.size / 2 / lp.mm], cmap='jet')
            plt.title(f'Intensity profile {i + 1}\n'
                      f'center_x={params[0]:.2f} mm, center_y={params[1]:.2f} mm\n'
                      f'sigma_x={params[2]:.2f} mm, sigma_y={params[3]:.2f}, angle={params[4]:.2f} rad')
            plt.colorbar(label='Intensity')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def load_npz_to_dataframe(dataset_name):
        """
        Carga un dataset .npz y lo convierte en un DataFrame de Pandas.
        """
        data = np.load(dataset_name, allow_pickle=True)

        # Extraer datos
        intensity_matrices = data['intensity_matrices']
        parameters = data['parameters']

        return intensity_matrices, parameters
