import numpy as np

# Створення матриці коефіцієнтів
A = np.array([[-(2 + 1), 1.5, 1.2, 0.5],
              [2, -(1.5 + 0.02), 0, 0],
              [1, 0, -(1.2 + 0.03), 0],
              [1, 1, 1, 1]])

# Створення вектора вільних членів
b = np.array([0, 0, 0, 1])

# Розв'язання системи рівнянь
p = np.linalg.solve(A, b)

# Виведення результату
print("Граничні ймовірності pi:", p)