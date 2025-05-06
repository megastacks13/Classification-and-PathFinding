import numpy as np
import math
from collections import defaultdict


class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = []
        self.labels = []
        self.classes = []

    def load_data(self):
        with open(self.filepath, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split(',')
                    features = list(map(float, parts[:4]))
                    label = parts[4].strip()

                    self.data.append(features)
                    self.labels.append(label)

                    if label not in self.classes:
                        self.classes.append(label)

        self.data = np.array(self.data)
        return self.data, self.labels, self.classes


class FuzzyCMeans:
    def __init__(self, b=2, tolerance=0.2, max_iter=100):
        # Definimos una variable para el número de clusteres
        self.C = None
        # Definimos una variable para el número de muestras sin definir por el momento
        self.N = None
        # Definimos una variable para los centros de los clusteres
        self.V = None
        # Y para la matriz de pertenencia
        self.U = None

        # Valores nominales de las clases
        self.classes = []

        # Finalmente creamos variables para ponderación, tolerancia y límite de iteraciones
        self.b = b
        self.tolerance = tolerance
        self.max_iter = max_iter


    def _initialize_centers(self):
        # Inicialización con los valores sugeridos en el apéndice
        self.V = np.array([[4.6, 3.0, 4.0, 0.0],
                           [6.8, 3.4, 4.6, 0.7]])

    def _calculate_membership(self, data):
        # Calcular distancias euclideas entre cada punto y cada centro
        distances = np.zeros((self.N, self.C))
        for i in range(self.C):
            distances[:, i] = np.linalg.norm(data - self.V[i], axis=1)

        # Evitar división por cero
        distances[distances == 0] = 1e-10

        # Calcular matriz de pertenencia U
        power = 2 / (self.b - 1)
        U = np.zeros((self.N, self.C))

        for i in range(self.C):
            for j in range(self.C):
                U[:, i] += (distances[:, i] / distances[:, j]) ** power

        U = 1 / U
        return U

    def _update_centers(self, data, U):
        # Actualizar centros usando la matriz de pertenencia
        U_b = U ** self.b
        new_V = np.zeros_like(self.V)

        for i in range(self.C):
            numerator = np.sum(U_b[:, i:i + 1] * data, axis=0)
            denominator = np.sum(U_b[:, i])
            new_V[i] = numerator / denominator

        return new_V

    def _fit(self, data):
        self.N = data.shape[0]
        self._initialize_centers()

        for iteration in range(self.max_iter):
            old_V = self.V.copy()
            self.U = self._calculate_membership(data)
            self.V = self._update_centers(data, self.U)

            # Verificar criterio de convergencia
            if np.linalg.norm(self.V - old_V) < self.tolerance:
                break

    def predict(self, filepath):
        loader = DataLoader(filepath)
        predict_data, labels, real_data_class = loader.load_data()

        # Calcular distancias al centroide
        distances = np.array([np.linalg.norm(predict_data - v) for v in self.V])

        # Calcular pertenencia para la nueva muestra
        membership = 1 / (distances ** (2 / (self.b - 1)))
        membership /= np.sum(membership)

        # Devolver el cluster con mayor pertenencia
        max_val = np.argmax(membership)
        predicted_class = self.classes[max_val]
        return max_val, predicted_class, real_data_class[0], distances, predict_data[0]

    def algorithm(self, filepath):
        # Cargar datos
        loader = DataLoader(filepath)
        data, labels, classes = loader.load_data()

        # Establecemos los valores de C y N
        self.C = len(classes)
        self.N = len(data)

        # Almacenamos la lista de las clases para la predicción
        self.classes = classes

        # Ajustar el modelo
        self._fit(data)

        # Devolver los centros calculados
        return self.V


class BayesClassifier:
    def __init__(self):
        self.classes = None
        self.class_priors = None
        self.means = None
        self.variances = None

    def _fit(self, data, labels):
        self.classes = np.unique(labels)
        n_classes = len(self.classes)
        n_features = data.shape[1]

        self.means = np.zeros((n_classes, n_features))
        self.variances = np.zeros((n_classes, n_features))
        self.class_priors = np.zeros(n_classes)

        for i, c in enumerate(self.classes):
            class_data = data[labels == c]
            self.means[i, :] = np.mean(class_data, axis=0)
            self.variances[i, :] = np.var(class_data, axis=0)
            self.class_priors[i] = class_data.shape[0] / data.shape[0]

    def _gaussian_prob(self, x, mean, var):
        # Función de densidad de probabilidad gaussiana
        exponent = np.exp(-((x - mean) ** 2) / (2 * var))
        return (1 / np.sqrt(2 * np.pi * var)) * exponent

    def predict(self, filepath):
        loader = DataLoader(filepath)
        predict_data, labels, real_data_class = loader.load_data()
        posteriors = []

        for i, c in enumerate(self.classes):
            prior = np.log(self.class_priors[i])
            likelihood = np.sum(np.log(self._gaussian_prob(predict_data, self.means[i], self.variances[i])))
            posterior = prior + likelihood
            posteriors.append(posterior)

        # Devolver el cluster con mayor pertenencia
        max_val = np.argmax(posteriors)
        predicted_class = self.classes[max_val]
        return max_val, predicted_class, real_data_class[0], posteriors, predict_data[0]


    def algorithm(self, filepath):
        # Cargar datos
        loader = DataLoader(filepath)
        data, labels, classes = loader.load_data()
        labels = np.array(labels)

        # Almacenamos la lista de las clases para la predicción
        self.classes = classes

        # Ajustar el modelo
        self._fit(data, labels)

        return self.means, self.variances, self.class_priors


class LloydAlgorithm:

    def __init__(self, n_clusters=2, learning_rate=0.1, tolerance=1e-10, max_iter=10):
        self.n_clusters = n_clusters
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.centers = None
        self.classes = []

    def _initialize_centers(self, data):
        # Inicialización con los valores sugeridos en el apéndice
        if self.n_clusters == 2:
            self.centers = np.array([[4.6, 3.0, 4.0, 0.0],
                                     [6.8, 3.4, 4.6, 0.7]])
        else:
            # Inicialización aleatoria si no son 2 clusters
            idx = np.random.choice(data.shape[0], self.n_clusters, replace=False)
            self.centers = data[idx]

    def _assign_clusters(self, data):
        distances = np.zeros((data.shape[0], self.n_clusters))

        for i in range(self.n_clusters):
            distances[:, i] = np.linalg.norm(data - self.centers[i], axis=1)

        return np.argmin(distances, axis=1)

    def _update_centers(self, data, labels):
        new_centers = np.zeros_like(self.centers)

        for i in range(self.n_clusters):
            cluster_data = data[labels == i]
            if len(cluster_data) > 0:
                # Actualización con razón de aprendizaje
                centroid = np.mean(cluster_data, axis=0)
                new_centers[i] = self.centers[i] + self.learning_rate * (centroid - self.centers[i])
            else:
                new_centers[i] = self.centers[i]

        return new_centers

    def _fit(self, data):
        self._initialize_centers(data)

        for iteration in range(self.max_iter):
            old_centers = self.centers.copy()
            labels = self._assign_clusters(data)
            self.centers = self._update_centers(data, labels)

            # Criterio de parada
            if np.linalg.norm(self.centers - old_centers) < self.tolerance:
                break

    def predict(self, filepath):
        # Cargamos los datos
        loader = DataLoader(filepath)
        predict_data, labels, real_data_class = loader.load_data()

        # Comprobamos que predice el modelo
        distances = np.array([np.linalg.norm(predict_data - center) for center in self.centers])

        # Sacamos el valor mínimo (correspondiente a la solución) y su nombre
        min_value = np.argmin(distances)
        cluster_value = self.classes[min_value]
        return min_value, cluster_value, real_data_class[0], distances, predict_data[0]

    def algorithm(self, filepath):
        # Cargar datos
        loader = DataLoader(filepath)
        data, labels, classes = loader.load_data()
        labels = np.array(labels)

        # Almacenamos la lista de las clases para la predicción
        self.classes = classes

        # Ajustar el modelo
        self._fit(data)

        return self.centers

def run_fuzzy_means():
    print("========================> Agrupamiento borroso (K-medias) ========>")

    # Crear instancia del algoritmo con valores por defecto
    fcm = FuzzyCMeans()

    # Cargar datos y ejecutar algoritmo
    centers = fcm.algorithm('Iris2Clases.txt')
    print("\nModelo entrenado con éxito")
    print("\nCentros finales:")
    print(f"\t Clúster 0: {centers[0]}")
    print(f"\t Clúster 1: {centers[1]}")

    # Predecir una muestra de prueba 1
    cluster, cluster_name, real_class, distances, data = fcm.predict('TestIris01.txt')
    print(f"\nPrediciendo datos {data}:")
    print(f"\tDistancia al clúster 0: {distances[0]}")
    print(f"\tDistancia al clúster 1: {distances[1]}")
    print(f"\tClúster a menor distancia: clúster {cluster}")
    print(f"\nLa muestra 1 pertenece a la clase '{cluster_name}' y debía ser de clase '{real_class}'.")
    print()

    # Predecir una muestra de prueba 2
    cluster, cluster_name, real_class, distances, data = fcm.predict('TestIris02.txt')
    print(f"\nPrediciendo datos {data}:")
    print(f"\tDistancia al clúster 0: {distances[0]}")
    print(f"\tDistancia al clúster 1: {distances[1]}")
    print(f"\tClúster a menor distancia: clúster {cluster}")
    print(f"\nLa muestra 2 pertenece a la clase '{cluster_name}' y debía ser de clase '{real_class}'.")
    print()

    # Predecir una muestra de prueba 3
    cluster, cluster_name, real_class, distances, data = fcm.predict('TestIris03.txt')
    print(f"\nPrediciendo datos {data}:")
    print(f"\tDistancia al clúster 0: {distances[0]}")
    print(f"\tDistancia al clúster 1: {distances[1]}")
    print(f"\tClúster a menor distancia: clúster {cluster}")
    print(f"\nLa muestra 3 pertenece a la clase '{cluster_name}' y debía ser de clase '{real_class}'.")
    print()

def run_bayes_parametric_estimation():
    print("========================> Estimación paramétrica (Bayes) ========>")
    # Crear instancia del algoritmo con valores por defecto
    bcf = BayesClassifier()

    means, variances, class_priors = bcf.algorithm('Iris2Clases.txt')
    print("\nModelo entrenado con éxito")
    print("\nMedias finales:")
    print(f"\tClúster 0: {means[0]}")
    print(f"\tClúster 1: {means[1]}")

    print("\nVarianzas finales:")
    print(f"\tClúster 0: {variances[0]}")
    print(f"\tClúster 1: {variances[1]}")

    # Predecir una muestra de prueba 1
    cluster, cluster_name, real_class, posteriors, data = bcf.predict('TestIris01.txt')
    print(f"\nPrediciendo datos {data}:")
    print(f"\tPosteriors: {posteriors}")
    print(f"\tPosición 'Posterior' máximo (equivalente al cluster correcto): clúster {cluster}")
    print(f"\nLa muestra 1 pertenece a la clase `{cluster_name}` y debía ser de clase '{real_class}'.")
    print()

    # Predecir una muestra de prueba 1
    cluster, cluster_name, real_class, posteriors, data = bcf.predict('TestIris02.txt')
    print(f"\nPrediciendo datos {data}:")
    print(f"\tPosteriors: {posteriors}")
    print(f"\tPosición 'Posterior' máximo (equivalente al cluster correcto): clúster {cluster}")
    print(f"\nLa muestra 2 pertenece a la clase `{cluster_name}` y debía ser de clase '{real_class}'.")
    print()

    # Predecir una muestra de prueba 1
    cluster, cluster_name, real_class, posteriors, data = bcf.predict('TestIris03.txt')
    print(f"\nPrediciendo datos {data}:")
    print(f"\tPosteriors: {posteriors}")
    print(f"\tPosición 'Posterior' máximo (equivalente al cluster correcto): clúster {cluster}")
    print(f"\nLa muestra 1 pertenece a la clase `{cluster_name}` y debía ser de clase '{real_class}'.")
    print()

def run_lloyd_algorithm():
    print("========================> Algoritmo de Lloyd ========>")
    # Crear instancia del algoritmo con valores por defecto
    lla = LloydAlgorithm()

    # Cargar datos y ejecutar algoritmo
    centers = lla.algorithm('Iris2Clases.txt')
    print("\nModelo entrenado con éxito")
    print("\nCentros finales:")
    print(f"\t Clúster 0: {centers[0]}")
    print(f"\t Clúster 1: {centers[1]}")

    # Predecir una muestra de prueba 1
    cluster, cluster_name, real_class, distances, data = lla.predict('TestIris01.txt')
    print(f"\nPrediciendo datos {data}:")
    print(f"\tDistancia al clúster 0: {distances[0]}")
    print(f"\tDistancia al clúster 1: {distances[1]}")
    print(f"\tClúster a menor distancia: clúster {cluster}")
    print(f"\nLa muestra 1 pertenece al cluster '{cluster_name}' y debía ser de clase '{real_class}'.")
    print()

    # Predecir una muestra de prueba 2
    cluster, cluster_name, real_class, distances, data = lla.predict('TestIris02.txt')
    print(f"\nPrediciendo datos {data}:")
    print(f"\tDistancia al clúster 0: {distances[0]}")
    print(f"\tDistancia al clúster 1: {distances[1]}")
    print(f"\tClúster a menor distancia: clúster {cluster}")
    print(f"\nLa muestra 2 pertenece al cluster '{cluster_name}' y debía ser de clase '{real_class}'.")
    print()

    # Predecir una muestra de prueba 3
    cluster, cluster_name, real_class, distances, data = lla.predict('TestIris03.txt')
    print(f"\nPrediciendo datos {data}:")
    print(f"\tDistancia al clúster 0: {distances[0]}")
    print(f"\tDistancia al clúster 1: {distances[1]}")
    print(f"\tClúster a menor distancia: clúster {cluster}")
    print(f"\nLa muestra 3 pertenece al cluster '{cluster_name}' y debía ser de clase '{real_class}'.")
    print()

if __name__ == "__main__":
     run_fuzzy_means()
     run_bayes_parametric_estimation()
     run_lloyd_algorithm()
