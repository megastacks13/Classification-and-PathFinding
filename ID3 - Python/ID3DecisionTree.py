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

def run_id3(attributes, examples, result_tree, key_positive, key_position:int, edge_name=None, parent=None):
    # Importamos la variable del contador
    global global_counter
    # Miramos si son todos los elementos iguales
    all_same, leaf_label = get_all_same_sign(examples, key_position)
    # De ser todos iguales creamos un nodo con el valor obtenido
    if all_same:
        if not any(result_tree.has_edge(parent, node) and node == leaf_label for node in result_tree.successors(parent)):
            # Le añadimos caracteres vacíos alrededor para que mantengan el mismo display, pero no se conecten los nodos
            # La variable global sirve para evitar repeticiones futuras
            leaf_label = '‎'*global_counter+leaf_label+'‎'*global_counter
            global_counter += 1
            # Añadimos el edge y le asignamos un nombre
            result_tree.add_edge(parent, leaf_label)
            result_tree[parent][leaf_label]['name'] = edge_name
        return
    # Al no ser todos iguales
    # Calculamos los méritos
    summary_dict = get_attributes_probabilities(attributes=attributes, examples=examples, key_positive=key_positive,
                                                key_position=key_position)

    # Creamos una lista iterable que contiene todos los valores salvo el valor key
    iterable = [attr for i, attr in enumerate(summary_dict) if i != key_position]
    # De ser key el único valor de la lista, devolvemos
    if not iterable:
        return

    # Tomamos el atributo del diccionario con menor valor y sacamos su índice en la lista actual
    lower_merit_attribute = min(iterable, key=lambda attr: summary_dict[attr]['merit'])
    index = attributes.index(lower_merit_attribute)
    # Añadimos el nodo al árbol
    result_tree.add_node(lower_merit_attribute)

    # En caso de que no tenga ningún padre, edge name es None, entonces no se añade edge
    if edge_name is not None:
        result_tree.add_edge(parent, lower_merit_attribute)
        result_tree[parent][lower_merit_attribute]['name'] = edge_name

    # Establecemos los nuevos atributos como todos los anteriores salvo aquel elegido
    new_attributes = [attr for i, attr in enumerate(attributes) if i != index]
    # Actualizamos el key position
    new_key_position = new_attributes.index(attributes[key_position])
    for option in summary_dict[lower_merit_attribute]['options']:
        # Hacemos lo mismo con los ejemplos
        new_examples = [row[:index] + row[index+1:] for row in examples if row[index] == option]
        # Y por cada ejemplo del tipo elegido, volvemos a calcular el id3
        run_id3(attributes=new_attributes, examples=new_examples, result_tree=result_tree,  key_positive=key_positive,
                key_position=new_key_position, edge_name=option, parent=lower_merit_attribute)

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
    return predict(built_tree=built_tree, attributes=index_names, sample=sample_value)

def get_posible_answers(index, values_matrix):
    answers = [row[index] for row in values_matrix[:]]
    return list(set(answers))

# Método de prediccion que itera por el árbol construido
def predict(built_tree, attributes, sample):
    # Sacamos el nodo padre
    current_node = list(built_tree.nodes)[0]  # Suponemos que el primer nodo es la raíz

    # Vamos comparando cualidades
    while current_node in attributes:  # Mientras no sea un nodo hoja
        attr_index = attributes.index(current_node)
        sample_value = sample[attr_index]

        # Buscar la rama que corresponde al valor del atributo
        found = False
        for child in built_tree.successors(current_node):
            if built_tree[current_node][child]['name'] == sample_value:
                current_node = child
                found = True
                break

        if not found:
            return "No se puede determinar la clase (valor desconocido en el árbol)"


    # El nodo actual debe ser una hoja con la predicción
    # Aprovechamos y limpiamos la "chapuza" de antes
    return current_node.replace('‎', '')

# Esto es un código que te permite correrlo en la terminal si así lo gustas
if __name__ == '__main__':
    # Leemos atributos y los ejemplos proporcionados (material de entreno)
    attributes_names = read_file("AtributosJuego.txt", ",")
    example_names = read_file("Juego.txt", ",")
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