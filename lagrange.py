import numpy as np
import matplotlib.pyplot as plt

def lagrange_interpolation(x, y, xi):
    result = 0
    for j in range(len(y)):
        term = y[j]
        for k in range(len(x)):
            if k != j:
                term = term * (xi - x[k]) / (x[j] - x[k])
        result += term
    return result

def lagrange_formula(xi, yi):
    lagrange_formula = "Ln(x) = "
    for i in range(len(xi)):
        lagrange_formula += f"({yi[i]:+})"  
        for j in range(len(xi)):
            if j != i:
                lagrange_formula += f" * (x - {xi[j]}) / ({xi[i]} - {xi[j]})"
        if i != len(xi) - 1:
            lagrange_formula += " + "
    print("Формула многочлена Лагранжа:")
    print(lagrange_formula)

# Задана таблиця значень функції
xi1 = np.array([-1, 0, 1, 2])
yi1 = np.array([5, -11, -3, 23])

# Задані точки
interpolation_points = np.array([-2, -0.5, 0.5, 1.5])

# Виконуємо інтерполяцію та обчислюємо значення функції в заданих точках
interpolated_values = [lagrange_interpolation(xi1, yi1, xi) for xi in interpolation_points]

print("Значення функції в заданих точках:")
for xi, yi in zip(interpolation_points, interpolated_values):
    print(f"f({xi}) = {yi}")

lagrange_formula(xi1, yi1)

x_range = np.linspace(-2, max(xi1), 1000)
y_interpolated = [lagrange_interpolation(xi1, yi1, xi) for xi in x_range]

plt.plot(x_range, y_interpolated, label="Ln(x)")
plt.scatter(xi1, yi1, color='red', label="Таблиця значень")
plt.scatter(interpolation_points, interpolated_values, color='green', label="Задані точки")
plt.legend()
plt.title("Інтерполяційний многочлен Лагранжа та графік функції")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
