import matplotlib.pyplot as plt
import numpy as np


class ClusterVisualizer:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        # Establecemos los colores para cada uno de los clústeres
        self.colors = ['red', 'blue']
        self.marker_size = 100
        self.sample_marker_size = 200

    def visualize(self, algorithm_name, centers, data, sample_data, sample_cluster, features_idx=(0, 1)):
        # Escogemos los atributos a visualizar
        x_idx, y_idx = features_idx
        centers_2d = centers[:, [x_idx, y_idx]]
        data_2d = data[:, [x_idx, y_idx]]
        sample_2d = sample_data[[x_idx, y_idx]]

        # Vaciamos el plot anterior
        self.ax.clear()

        # Mostramos todos los datos del color del centro más cercano
        distances = np.array([np.linalg.norm(data_2d - center, axis=1) for center in centers_2d])
        labels = np.argmin(distances, axis=0)

        for i, center in enumerate(centers_2d):
            # Plot de los datos
            cluster_data = data_2d[labels == i]
            self.ax.scatter(cluster_data[:, 0], cluster_data[:, 1],
                            c=self.colors[i], s=self.marker_size,
                            alpha=0.5, label=f'Cluster {i}')

            # Plot del centro con una X
            self.ax.scatter(center[0], center[1], c=self.colors[i],
                            s=self.marker_size * 2, marker='X', edgecolor='black')

        # Marcamos más fuerte el color de la muestra
        self.ax.scatter(sample_2d[0], sample_2d[1], c=self.colors[sample_cluster],
                        s=self.sample_marker_size, edgecolor='black', linewidth=2,
                        label='Sample Point')

        # Ver detalles
        self.ax.set_title(f'{algorithm_name} - Cluster Visualization')
        self.ax.set_xlabel(f'Feature {x_idx + 1}')
        self.ax.set_ylabel(f'Feature {y_idx + 1}')
        self.ax.legend()
        plt.tight_layout()

    def show(self):
        plt.show()