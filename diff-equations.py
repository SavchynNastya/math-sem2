# Перед запуском: pip install numpy matplotlib pandas

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Задане диференціальне рівняння
def differential_equation(x, y):
    return y - x

# Методи числового розв'язку

def euler_method(x, y, h):
    return y + h * differential_equation(x, y)

def euler_cauchy_method(x, y, h):
    return y + h * differential_equation(x + h/2, y + h/2 * differential_equation(x, y))

def improved_euler_method(x, y, h):
    k1 = differential_equation(x, y)
    k2 = differential_equation(x + h, y + h * k1)
    return y + h/2 * (k1 + k2)

def runge_kutta_method(x, y, h):
    k1 = h * differential_equation(x, y)
    print(k1)
    k2 = h * differential_equation(x + h/2, y + k1/2)
    print(k2)
    k3 = h * differential_equation(x + h/2, y + k2/2)
    k4 = h * differential_equation(x + h, y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

# Задаємо початкові умови
x0, y0 = 0, 1.5
h = 0.1
end_x = 0.1

# Обчислюємо значення для кожного методу
x_values = np.linspace(x0, end_x, int((end_x - x0) / h) + 1)
y_euler, y_euler_cauchy, y_improved_euler, y_runge_kutta = [y0], [y0], [y0], [y0]

for x in x_values[1:]:
    y_euler.append(euler_method(x, y_euler[-1], h))
    y_euler_cauchy.append(euler_cauchy_method(x, y_euler_cauchy[-1], h))
    y_improved_euler.append(improved_euler_method(x, y_improved_euler[-1], h))
    y_runge_kutta.append(runge_kutta_method(x, y_runge_kutta[-1], h))

# # Точний розв'язок
# y_exact = [0.8, 0.905575, 1.02550, 1.161250, 1.314480, 1.486980, 1.680740, 1.897980, 2.141110, 2.412820, 2.716060]

# # Збираємо результати в таблицю
# df = pd.DataFrame({
#     'x': x_values,
#     'Euler Method': y_euler,
#     'Euler-Cauchy Method': y_euler_cauchy,
#     'Improved Euler Method': y_improved_euler,
#     'Runge-Kutta Method': y_runge_kutta,
#     'Exact Analytical Solution': y_exact
# })

# # Виводимо таблицю
# print(df)

# # Побудова графіків
# plt.plot(x_values, y_euler, label='Euler Method')
# plt.plot(x_values, y_euler_cauchy, label='Euler-Cauchy Method')
# plt.plot(x_values, y_improved_euler, label='Improved Euler Method')
# plt.plot(x_values, y_runge_kutta, label='Runge-Kutta Method')
# plt.plot([0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6], y_exact, label='Exact analytical', linestyle='--')

# plt.title("Numerical Methods for Differential Equation")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend()
# plt.grid(True)
# plt.show()
