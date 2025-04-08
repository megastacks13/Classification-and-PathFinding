# **********************************************
# **Author:** Jaime Alonso Fernández
# **Date:** - 27-03-2025
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
import math # Para los logaritmos del mérito
from collections import Counter

# Esto es una constante para un "engaño" que he tenido que llevar a cabo para el plot de los nodos
global_counter = 0

# Método sencillo que lee desde el archivo proporcionado el texto de entreno
def read_file(filepath, separator) -> []:
    # Usamos un try catch por posibles errores (gracias open)
    try:
        file_values = []
        # Abrimos de lectura
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


def run_id3(attributes, examples, result_tree, key_positive, key_position: int, edge_name=None, parent=None):
    global global_counter

    # Caso base: si no hay ejemplos, retornar
    if not examples:
        return

    # Verificar si todos los ejemplos tienen la misma clasificación
    all_same, leaf_label = get_all_same_sign(examples, key_position)
    if all_same:
        if parent is not None:  # Solo añadir hoja si hay un padre
            if not any(result_tree.has_edge(parent, node) and node == leaf_label for node in
                       result_tree.successors(parent)):
                leaf_label = f"{leaf_label}_{global_counter}"  # Mejor alternativa a caracteres especiales
                global_counter += 1
                result_tree.add_edge(parent, leaf_label)
                if edge_name is not None:
                    result_tree[parent][leaf_label]['name'] = edge_name
        return  # Importante: retornar aquí

    # Si no hay atributos para dividir (excepto el target), crear nodo hoja con mayoría
    if len(attributes) <= 1:
        majority_label = get_majority_label(examples, key_position)
        if parent is not None:
            result_tree.add_edge(parent, majority_label)
            if edge_name is not None:
                result_tree[parent][majority_label]['name'] = edge_name
        return

    # Calcular el mejor atributo para dividir
    summary_dict = get_attributes_probabilities(attributes=attributes, examples=examples,
                                                key_positive=key_positive, key_position=key_position)

    # Excluir el atributo target de la división
    iterable = [attr for attr in summary_dict if attr != attributes[key_position]]
    if not iterable:
        return

    lower_merit_attribute = min(iterable, key=lambda attr: summary_dict[attr]['merit'])
    new_node_name = f"{lower_merit_attribute}_{global_counter}"
    global_counter += 1
    index = attributes.index(lower_merit_attribute)

    # Añadir el nuevo nodo al árbol
    result_tree.add_node(new_node_name)
    if parent is not None and edge_name is not None:
        result_tree.add_edge(parent, new_node_name)
        result_tree[parent][new_node_name]['name'] = edge_name

    # Preparar para la recursión
    new_attributes = [attr for attr in attributes if attr != lower_merit_attribute]
    try:
        new_key_position = new_attributes.index(attributes[key_position])
    except ValueError:
        new_key_position = -1  # En caso de que el atributo target haya sido eliminado

    for option in summary_dict[lower_merit_attribute]['options']:
        new_examples = [row[:index] + row[index + 1:] for row in examples if row[index] == option]
        if not new_examples:
            # Añadir nodo hoja para opción sin ejemplos
            majority_label = get_majority_label(examples, key_position)
            leaf_name = f"{majority_label}_{global_counter}"
            global_counter += 1
            result_tree.add_edge(new_node_name, leaf_name)
            result_tree[new_node_name][leaf_name]['name'] = option
            continue

        run_id3(attributes=new_attributes, examples=new_examples, result_tree=result_tree,
                key_positive=key_positive, key_position=new_key_position,
                edge_name=option, parent=new_node_name)



def get_majority_label(examples, key_position):
    if not examples:
        return None

    # Contar frecuencia de cada etiqueta
    label_counts = {}
    for example in examples:
        label = example[key_position]
        label_counts[label] = label_counts.get(label, 0) + 1

    # Encontrar la etiqueta con mayor frecuencia
    majority_label = max(label_counts.items(), key=lambda x: x[1])[0]

    return majority_label

# Método para saber si los elementos de la tabla tienen como atributo objetivo el mismo
def get_all_same_sign(example, key_position):
    # Caso extremo de error donde example no existe (no se va a ha dar con el modelo proporcionado)
    if not example:
        exit(2)
    # Obtenemos el valor de la primera fila
    value_to_count = example[0][key_position]
    # Contamos cuantos de los valores concuerdan con el valor de la primera
    value_count = sum(1 for row in example if row[key_position] == value_to_count)
    # Y devolvemos si los contados son el mismo número que los totales
    return len(example) == value_count, value_to_count

# Método que calcula las variables p, n, r y mérito y almacena la última en el diccionario
def get_attributes_probabilities(attributes, examples, key_position:int, key_positive):
    result = {}
    # Por cada uno de los atributos de la lista creamos una entrada en el diccionario
    for i, attribute in enumerate(attributes):
        result[attribute] = {'options': list(set(row[i] for row in examples))}
        merit = 0
        # Por cada entrada del atributo seleccionado
        for j, option in enumerate(result[attribute]['options']):
            # Esto sería 'n'
            total_element_count = sum(1 for row in examples if row[i] == option)
            # Contamos los valores positivos
            p_count = sum(1 for row in examples if row[i] == option and row[key_position] == key_positive)
            # Sacamos la fracción (p valor)
            p_value = p_count / total_element_count if total_element_count else 0
            # La inversa será entonces el n valor
            n_value = (total_element_count - p_count) / total_element_count if total_element_count else 0
            # Sacamos r
            r_value = total_element_count / len(examples)
            # Hacemos el cálculo del mérito
            merit += r_value * (-p_value * math.log2(p_value) if p_value else 0 - n_value * math.log2(n_value) if n_value else 0)
        # Guardamos el mérito en el diccionario
        result[attribute]['merit'] = merit
    # Devolvemos el diccionario
    return result


# Método que permite la visualización del árbol
def show_tree(graph):
    # Creamos una figura
    plt.figure(figsize=(10, 6))
    # Establecemos la posición de los nodos y su forma circular (dot)
    pos = nx.nx_agraph.graphviz_layout(graph, prog="dot")
    # Mostramos los nodos
    nx.draw(graph, pos, with_labels=True, node_color="skyblue", node_size=3000, font_size=10, edge_color="gray")
    # Etiquetamos los labels
    edge_labels = {(u, v): d['name'] for u, v, d in graph.edges(data=True)}
    # Mostramos los labels
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=9)
    # Mostramos los valores
    plt.show()

# Método que permite realizar predicciones en base a unos inputs
def ask_if_prediction_and_predict(index_names, pos_of_variable_to_predict, built_tree, examples_names):
    print('*************************************************')
    # Verificamos que el usuario quiera predecir algo
    make_predict = input('Deseas hacer una predicción en base al modelo ("si" para continuar): ').lower()
    if not make_predict == 'si':
        exit(0)

    print('Comienzo de proceso de añadido manual de datos a la muestra: ')
    sample_value = []
    # Preguntamos por cada elemento menos por aquel a predecir
    for i,name in enumerate(index_names):
        if i == pos_of_variable_to_predict:
            continue
        # Extraemos del dataset las opciones para verificar la respuesta
        answers = get_posible_answers(i, examples_names)
        nuevo_valor = None
        while nuevo_valor not in answers:
            nuevo_valor = input(f'\tValor para la variable {name} ({answers}): ')
        sample_value.append(nuevo_valor)
    # Devolvemos la prediccion
    return predict(built_tree=built_tree, attributes=index_names, sample=sample_value, key_position=k_pos, example=examples_names)

def get_posible_answers(index, values_matrix):
    answers = [row[index] for row in values_matrix[:]]
    return list(set(answers))

# Método de prediccion que itera por el árbol construido
def predict(built_tree, attributes, sample, key_position, example):
    # Obtenemos el nodo raíz (el que no tiene predecesores)
    root_nodes = [node for node in built_tree.nodes() if built_tree.in_degree(node) == 0]
    if not root_nodes:
        return "Árbol no válido: no se encontró nodo raíz"

    current_node = root_nodes[0]

    # Obtenemos todas las posibles etiquetas finales
    final_values = set(row[key_position] for row in example)

    while True:
        # Si llegamos a un nodo que es una etiqueta final (limpiamos el sufijo numérico si existe)
        clean_node = current_node.split('_')[0]  # Eliminamos el sufijo _numero si existe
        if clean_node in final_values:
            return clean_node

        # Verificamos si el nodo actual es un atributo que tenemos en nuestra muestra
        attr_name = current_node.split('_')[0]  # Eliminamos el sufijo numérico para comparar
        if attr_name not in attributes:
            return "No se puede determinar la clase (nodo no reconocido)"

        # Obtenemos el índice del atributo en la lista original
        try:
            attr_index = attributes.index(attr_name)
        except ValueError:
            return "No se puede determinar la clase (atributo no encontrado)"

        # Verificamos que el índice sea válido para la muestra
        if attr_index >= len(sample):
            return "No se puede determinar la clase (índice de atributo inválido)"

        sample_value = sample[attr_index]

        # Buscamos la rama que corresponde al valor del atributo
        found = False
        for child in built_tree.successors(current_node):
            if built_tree[current_node][child]['name'] == sample_value:
                current_node = child
                found = True
                break

        if not found:
            # Si no encontramos una rama, devolvemos la clase mayoritaria de los ejemplos
            majority_label = get_majority_label(example, key_position)
            return majority_label if majority_label else "No se puede determinar la clase"

# Esto es un código que te permite correrlo en la terminal si así lo gustas
if __name__ == '__main__':
    # Leemos atributos y los ejemplos proporcionados (material de entreno)
    attributes_names = read_file("AtributosJuego3.txt", ",")
    example_names = read_file("Juego3.txt", ",")
    pos_value = 'si' # Este es el valor que consideramos como bueno
    k_pos = len(example_names[0]) - 1 # Para el caso específico, es la última columna

    # Verificamos que hayan sido leídos correctamente
    if not attributes_names or not example_names:
        print('Ha habido un error leyendo los archivos de entreno.')
        exit(1)

    # Como todo ha salido bien, procedemos a crear el árbol
    tree = nx.DiGraph()
    # Y construímos el modelo id3
    run_id3(attributes=attributes_names, examples=example_names, result_tree=tree, key_positive=pos_value, key_position=k_pos)

    # Verificamos que todo se cree correctamente
    if tree.number_of_nodes() <= 0:
        print('Error de creación del árbol: árbol vacío')
        exit(1)
    # Si el id3 se creó correctamente, lo mostramos...
    show_tree(tree)

    # Y procedemos a preguntar las predicciones
    while True:
        prediction = ask_if_prediction_and_predict(index_names=attributes_names, pos_of_variable_to_predict=k_pos,
                                                   built_tree=tree, examples_names=example_names)
        print()
        print(f'El modelo ha predicho: {prediction}')