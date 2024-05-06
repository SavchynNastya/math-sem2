# import numpy as np
# import networkx as nx
# import matplotlib.pyplot as plt

# class MarkovChain:
#     def __init__(self, transition_matrix):
#         self.P = transition_matrix
#         self.initial_state = np.array([1, 0, 0, 0, 0])

#     def calculate_probabilities(self):
#         probabilities = [self.initial_state]
#         for i in range(5):
#             probabilities.append(np.dot(probabilities[-1], self.P))
#         return probabilities

#     def print_matrix(self):
#         print('Матриця переходів')
#         for row in self.P:
#             print(' ; '.join('{:.2f}'.format(cell) for cell in row))
#         print()

#     def print_probabilities(self, probabilities):
#         print("Початкові імовірності станів:")
#         for i in range(len(self.initial_state)):
#             print(f"p{i+1}(0) = {self.initial_state[i]}")
#         print()

#         for i in range(5):
#             print(f"Після {i+1}-го тесту:")
#             for j in range(len(probabilities[i])):
#                 print(f"p{j+1}({i+1}) = {probabilities[i+1][j]}")
#             print()

#     def check_results(self, probabilities):
#         p0 = self.initial_state
#         p5_calculated = np.dot(p0, np.linalg.matrix_power(self.P, 5))
#         if np.allclose(probabilities[5], p5_calculated):
#             print("\nРезультати перевірки співпадають: обчислені імовірності відповідають розрахованим.")
#         else:
#             print("\nРезультати перевірки не співпадають: обчислені імовірності не відповідають розрахованим.")

#     def draw_transition_graph(self):
#         G = nx.DiGraph()
#         for i in range(5):
#             for j in range(5):
#                 if self.P[i][j] > 0:
#                     G.add_edge(f'S{i+1}', f'S{j+1}', weight=self.P[i][j])

#         pos = nx.spring_layout(G)
#         nx.draw(G, pos, with_labels=True, node_size=2000, node_color='skyblue', font_size=10, font_weight='bold')
#         labels = nx.get_edge_attributes(G, 'weight')
#         nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
#         plt.title('Граф переходів')
#         plt.show()

# # Оголошення матриці переходів
# P = np.array([[0, 0.20, 0.15, 0.10, 0.05],
#               [0, 0, 0.40, 0.20, 0.10],
#               [0, 0, 0, 0.35, 0.20],
#               [0, 0, 0, 0, 0.50],
#               [0, 0, 0, 0, 0]])

# # Створення об'єкту класу MarkovChain
# chain = MarkovChain(P)

# # Вивід матриці переходів
# chain.print_matrix()

# # Розрахунок імовірностей
# probabilities = chain.calculate_probabilities()

# # Вивід результатів
# chain.print_probabilities(probabilities)

# # Перевірка результатів
# chain.check_results(probabilities)

# # Побудова графа переходів
# chain.draw_transition_graph()

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class MarkovChain:
    def __init__(self, transition_matrix):
        self.P = transition_matrix
        self.initial_state = np.array([1, 0, 0, 0, 0])

    def calculate_probabilities(self):
        probabilities = [self.initial_state]
        for i in range(5):
            probabilities.append(np.dot(probabilities[-1], self.P))
        return probabilities

    def print_matrix(self):
        print('Матриця переходів')
        for row in self.P:
            print(' ; '.join('{:.2f}'.format(cell) for cell in row))
        print()

    def print_probabilities(self, probabilities):
        print("Початкові імовірності станів:")
        for i in range(len(self.initial_state)):
            print(f"p{i+1}(0) = {self.initial_state[i]}")
        print()

        for i in range(5):
            print(f"Після {i+1}-го тесту:")
            for j in range(len(probabilities[i])):
                print(f"p{j+1}({i+1}) = {probabilities[i+1][j]}")
            print()

    def check_results(self, probabilities):
        p0 = self.initial_state
        p5_calculated = np.dot(p0, np.linalg.matrix_power(self.P, 5))
        if np.allclose(probabilities[5], p5_calculated):
            print("\nРезультати перевірки співпадають: обчислені імовірності відповідають розрахованим.")
        else:
            print("\nРезультати перевірки не співпадають: обчислені імовірності не відповідають розрахованим.")

    def draw_transition_graph(self):
        G = nx.DiGraph()
        for i in range(5):
            for j in range(5):
                if self.P[i][j] > 0:
                    G.add_edge(f'S{i+1}', f'S{j+1}', weight=self.P[i][j])

        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=2000, node_color='skyblue', font_size=10, font_weight='bold')
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        plt.title('Граф переходів')
        plt.show()

    def validate_transition_matrix(self):
        row_sums = np.sum(self.P, axis=1)
        for i, row_sum in enumerate(row_sums):
            if not np.isclose(row_sum, 1.0):
                print(f"Помилка: Сума елементів у рядку {i+1} не дорівнює 1.")
                # Виправлення: Обчислення елементів на головній діагоналі як 1 - сума елементів у кожному рядку
                self.P[i][i] = 1.0 - row_sum

# Оголошення матриці переходів
P = np.array([[0, 0.20, 0.15, 0.10, 0.05],
              [0, 0, 0.40, 0.20, 0.10],
              [0, 0, 0, 0.35, 0.20],
              [0, 0, 0, 0, 0.50],
              [0, 0, 0, 0, 0]])

# Створення об'єкту класу MarkovChain
chain = MarkovChain(P)

# Перевірка матриці переходів на коректність
chain.validate_transition_matrix()

# Вивід матриці переходів
chain.print_matrix()

# Розрахунок імовірностей
probabilities = chain.calculate_probabilities()

# Вивід результатів
chain.print_probabilities(probabilities)

# Перевірка результатів
chain.check_results(probabilities)

# Побудова графа переходів
chain.draw_transition_graph()
