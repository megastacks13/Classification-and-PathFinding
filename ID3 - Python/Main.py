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
import math

total_si_no = 0

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


def run_id3(attributes:[], examples: [], tree, edge_name=None, parent=None):
    global total_si_no
    all_same, label = get_all_same_sign(examples)
    if all_same:
        leaf_label = "si" if examples[0][-1] == "si" else "no"
        leaf_label =total_si_no*' ' + leaf_label + total_si_no * ' '
        total_si_no += 1
        tree.add_edge(parent, leaf_label)
        tree[parent][leaf_label]['name'] = edge_name
        return

    summary_dict = get_attributes_probabilities(attributes, examples)

    iterable = [attr for attr in summary_dict if attr != 'Jugar']
    if not iterable:
        return
    lower_merit_attribute = min(iterable, key=lambda attr: summary_dict[attr]['merit'])
    index = attributes.index(lower_merit_attribute)

    tree.add_node(lower_merit_attribute)

    if edge_name is not None:
        tree.add_edge(parent, lower_merit_attribute)
        tree[parent][lower_merit_attribute]['name'] = edge_name

    new_attributes = [attr for i, attr in enumerate(attributes) if i != index]
    for option in summary_dict[lower_merit_attribute]['options']:
        new_examples = [row[:index] + row[index+1:] for row in examples if row[index] == option]
        run_id3(new_attributes, new_examples, tree, option, lower_merit_attribute)


def get_all_same_sign(example:[]) -> (bool, str):
    if not example:
        return False, 'no'
    value_to_count = example[0][-1]
    value_count = sum(1 for row in example if row[-1] == value_to_count)
    return len(example) == value_count, value_count

def safe_log2(x):
    return math.log2(x) if x > 0 else 0

# Creamos un set que permite obtener el número de respuestas únicas para cada atributo
def get_attributes_probabilities(attributes: [], examples: [[]]) -> dict:
    result = {}
    for i, attribute in enumerate(attributes):
        result[attribute] = {}
        result[attribute]['options'] = list(set(row[i] for row in examples))
        result[attribute]['P_values'] = {}
        result[attribute]['N_values'] = {}
        result[attribute]['R_values'] = {}

        merit = 0
        for j, option in enumerate(result[attribute]['options']):
            positive_value_name = f'P{j}'
            negative_value_name = f'N{j}'
            r_value_name = f'R{j}'
            information_name = f'information'

            total_element_count = sum(1 for row in examples if row[i] == option)
            p_count = sum(1 for row in examples if row[i] == option and row[-1] == 'si')

            p_value = p_count / total_element_count
            n_value = (total_element_count - p_count) / total_element_count
            r_value = total_element_count / len(examples)

            merit += r_value * (-p_value * safe_log2(p_value) - n_value * safe_log2(n_value))

            result[attribute]['P_values'][positive_value_name] = p_value
            result[attribute]['N_values'][negative_value_name] = n_value
            result[attribute]['R_values'][r_value_name] = r_value

        result[attribute]['merit'] = merit
    return result

def show_tree(graph: nx.DiGraph):
    plt.figure(figsize=(10, 6))

    # Usamos Graphviz para darle forma de árbol
    pos = nx.nx_agraph.graphviz_layout(graph, prog="dot")

    # Dibujamos el grafo con nodos y aristas
    nx.draw(graph, pos, with_labels=True, node_color="skyblue", node_size=3000, font_size=10, edge_color="gray")

    # Agregamos etiquetas a las aristas
    edge_labels = {(u, v): d['name'] for u, v, d in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=9)

    plt.show()


if __name__ == '__main__':
    print("I am main")
    attributes_names = read_file("AtributosJuego.txt", ",")
    example_names = read_file("Juego.txt", ",")

    tree = nx.DiGraph()
    run_id3(attributes_names, example_names, tree=tree)

    if tree.number_of_nodes() > 0:  # Solo mostrar si hay nodos en el árbolE
        for branch in tree.edges(data=True):
            print(branch)
        show_tree(tree)
    else:
        print("El árbol está vacío, no se mostrará el gráfico.")