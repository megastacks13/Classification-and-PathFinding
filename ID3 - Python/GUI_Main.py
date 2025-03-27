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

import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Aquí importamos la lógica construida en el otro archivo
from ID3DecisionTree import *


class DecisionTreeApp:
    def __init__(self, _root):
        self.entry_values = None
        self.tree = None
        self.k_pos = None
        self.pos_value = None
        self.example_names = None
        self.attributes_names = None
        self.root = _root
        self.root.title("Árbol de Decisión ID3")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)

        # Configurar estilo
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
        self.style.configure('TButton', font=('Arial', 10), padding=5)
        self.style.configure('Header.TLabel', font=('Arial', 12, 'bold'))

        # Contenedor principal
        self.main_frame = ttk.Frame(_root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Panel izquierdo (controles)
        self.control_frame = ttk.LabelFrame(self.main_frame, text="Configuración", padding=10)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # Panel derecho (visualización)
        self.display_frame = ttk.LabelFrame(self.main_frame, text="Visualización del Árbol", padding=10)
        self.display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Sección de entrada de datos
        ttk.Label(self.control_frame, text="Archivos de datos necesarios:", style='Header.TLabel').pack(anchor=tk.W,
                                                                                                        pady=(0, 10))

        # Archivo de atributos
        attr_frame = ttk.Frame(self.control_frame)
        attr_frame.pack(fill=tk.X, pady=5)
        ttk.Label(attr_frame, text="Archivo de atributos:").pack(anchor=tk.W)
        self.entry_attributes = ttk.Entry(attr_frame, width=30)
        self.entry_attributes.pack(fill=tk.X, pady=2)
        self.entry_attributes.insert(0, "AtributosJuego.txt")

        # Archivo de ejemplos
        example_frame = ttk.Frame(self.control_frame)
        example_frame.pack(fill=tk.X, pady=5)
        ttk.Label(example_frame, text="Archivo de ejemplos:").pack(anchor=tk.W)
        self.entry_examples = ttk.Entry(example_frame, width=30)
        self.entry_examples.pack(fill=tk.X, pady=2)
        self.entry_examples.insert(0, "Juego.txt")

        # Botón para generar árbol
        self.btn_generate_tree = ttk.Button(
            self.control_frame,
            text="Generar Árbol de Decisión",
            command=self.generate_tree,
            style='TButton'
        )
        self.btn_generate_tree.pack(pady=10, fill=tk.X)

        # Separador
        ttk.Separator(self.control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # Sección de predicción
        ttk.Label(self.control_frame, text="Realizar Predicción:", style='Header.TLabel').pack(anchor=tk.W,
                                                                                               pady=(0, 10))

        self.prediction_frame = ttk.Frame(self.control_frame)
        self.prediction_frame.pack(fill=tk.BOTH, expand=True)

        self.prediction_entries = []
        self.entry_values = {}

        # Botón de predicción (inicialmente deshabilitado)
        self.prediction_btn = ttk.Button(
            self.control_frame,
            text="Predecir",
            command=self.predict,
            state=tk.DISABLED
        )
        self.prediction_btn.pack(pady=5, fill=tk.X)

        # Área de visualización del árbol
        self.canvas_frame = ttk.Frame(self.display_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        # Barra de estado
        self.status_bar = ttk.Label(
            self.display_frame,
            text="Listo",
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding=2
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def generate_tree(self):
        attributes_file = self.entry_attributes.get()
        examples_file = self.entry_examples.get()

        self.status_bar.config(text="Procesando archivos...")
        self.root.update()

        try:
            self.attributes_names = read_file(attributes_file, ",")
            self.example_names = read_file(examples_file, ",")

            if not self.attributes_names or not self.example_names:
                messagebox.showerror("Error", "No se pudieron leer los archivos correctamente.")
                self.status_bar.config(text="Error al leer archivos")
                return

            self.pos_value = 'si'  # Valor positivo a considerar
            self.k_pos = len(self.example_names[0]) - 1

            self.tree = nx.DiGraph()
            run_id3(
                attributes=self.attributes_names,
                examples=self.example_names,
                result_tree=self.tree,
                key_positive=self.pos_value,
                key_position=self.k_pos
            )

            if self.tree.number_of_nodes() <= 0:
                messagebox.showerror("Error", "Error al crear el árbol.")
                self.status_bar.config(text="Error al crear árbol")
                return

            self.display_tree()
            self.create_prediction_inputs()
            self.prediction_btn.config(state=tk.NORMAL)
            self.status_bar.config(text="Árbol generado correctamente")

        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error: {str(e)}")
            self.status_bar.config(text=f"Error: {str(e)}")

    def display_tree(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        pos = nx.nx_agraph.graphviz_layout(self.tree, prog="dot")

        # Configuración visual mejorada
        node_colors = ["skyblue" if "leaf" not in node else "lightgreen" for node in self.tree.nodes()]
        node_sizes = [1200 if "leaf" not in node else 800 for node in self.tree.nodes()]

        nx.draw(
            self.tree, pos,
            with_labels=True,
            node_color=node_colors,
            node_size=node_sizes,
            font_size=9,
            edge_color="gray",
            ax=ax,
            arrows=True
        )

        edge_labels = {(u, v): d['name'] for u, v, d in self.tree.edges(data=True)}
        nx.draw_networkx_edge_labels(
            self.tree,
            pos,
            edge_labels=edge_labels,
            font_size=8,
            ax=ax,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
        )

        plt.tight_layout()

        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_prediction_inputs(self):
        # Limpiar frame de predicción
        for widget in self.prediction_frame.winfo_children():
            widget.destroy()

        self.prediction_entries = []
        self.entry_values = {}

        for i, name in enumerate(self.attributes_names):
            if i == self.k_pos:
                continue

            answers = list(set([row[i] for row in self.example_names[:]]))

            frame = ttk.Frame(self.prediction_frame)
            frame.pack(fill=tk.X, pady=2)

            label = ttk.Label(frame, text=f"{name}:", width=20, anchor=tk.W)
            label.pack(side=tk.LEFT)

            # Usar Combobox en lugar de Entry
            combo = ttk.Combobox(frame, values=answers, state="readonly")
            combo.pack(side=tk.RIGHT, fill=tk.X, expand=True)
            combo.set(answers[0] if answers else "")  # Seleccionar primer valor por defecto

            self.prediction_entries.append(combo)
            self.entry_values[name] = combo

    def predict(self):
        sample = []

        for i, name in enumerate(self.attributes_names):
            if i == self.k_pos:
                continue
            value = self.entry_values[name].get()
            if not value:
                messagebox.showwarning("Advertencia", f"Por favor ingrese un valor para {name}")
                return
            sample.append(value)

        try:
            prediction_value = predict(self.tree, self.attributes_names, sample)
            messagebox.showinfo(
                "Resultado de Predicción",
                f"El modelo ha predicho: {prediction_value}\n\n" +
                f"Atributos evaluados:\n{sample}"
            )
            self.status_bar.config(text=f"Predicción realizada: {prediction_value}")
        except Exception as e:
            messagebox.showerror("Error", f"Error al realizar la predicción: {str(e)}")
            self.status_bar.config(text=f"Error en predicción: {str(e)}")


if __name__ == '__main__':
    root = tk.Tk()
    app = DecisionTreeApp(root)
    root.mainloop()