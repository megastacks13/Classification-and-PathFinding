# **********************************************
# **Author:** Jaime Alonso Fernández
# **Date:** - 20-03-2025
#
# **Universidad Complutense de Madrid**
#
# **Grado en Ingeniería del Software (Plan 2019)**
#
# **Ingeniería del Conocimiento**
# **********************************************


# Importamos las librerías necesarias
import networkx as nx # Para el árbol
import matplotlib.pyplot as plt # Para el plot

def read_file(filepath, separator) -> []:
    try:
        file_values = []
        with open(filepath, 'r') as file:
            for line in file:
                # Borramos el fin de línea en caso de que lo tenga
                line = line.strip("\n")
                # Dividimos los textos
                split_line = line.split(separator)
                # Almacenamos la lista procesada
                file_values.append(split_line)

        # Si no hay elementos, devolvemos null
        if len(file_values) == 0:
            return None
        # Si solo tenemos una fila, devolvemos solo esa fila
        if len(file_values) == 1:
            return file_values[0]
        # Si no, devolvemos todos los valores
        return file_values
    except FileNotFoundError:
        # En caso de no existir el archivo, devolvemos null
        print("\033[91mFile not found: ", filepath, "\033[0m")

        return None


def run_id3(attributes:[], examples: []):
    pass


def build_tree() -> nx.DiGraph:
    graph = nx.DiGraph()

    # Agregar nodos y aristas (conexiones)
    graph.add_edges_from([
        ("Root", "Child 1"),
        ("Root", "Child 2"),
        ("Child 1", "Grandchild 1"),
        ("Child 1", "Grandchild 2"),
        ("Child 2", "Grandchild 3")
    ])

    return graph

def show_tree(graph: nx.DiGraph):
    plt.figure(figsize=(8, 5))
    pos = nx.spring_layout(graph)  # Distribución automática
    nx.draw(graph, pos, with_labels=True, node_color="skyblue", node_size=3000, font_size=10, edge_color="gray")
    plt.show()


if __name__ == '__main__':
    print("I am main")
    attributes_names = read_file("AtributosJuego.txt", ",")
    example_names = read_file("Juego.txt", ",")
    run_id3(attributes_names, example_names)
