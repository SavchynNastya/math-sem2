import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class OptimizationMethods:
    def __init__(self, f, a, b, epsilon, d1, d2, d3, d4):
        self.f = f
        self.a = a
        self.b = b
        self.epsilon = epsilon
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        self.d4 = d4

    def bisection_method(self):
        iterations = 0
        data = {'Ітерація k': [], 'ak': [], 'bk': [], 'xk(c)': [], '|bk-ak|': []}
        a, b = self.a, self.b
        while (b - a) >= self.epsilon:
            iterations += 1
            c = (a + b) / 2
            data['Ітерація k'].append(iterations)
            data['ak'].append(a)
            data['bk'].append(b)
            data['xk(c)'].append(c)
            data['|bk-ak|'].append(abs(b - a))
            y1 = self.f(c - self.epsilon, self.d1, self.d2, self.d3, self.d4)
            y2 = self.f(c + self.epsilon, self.d1, self.d2, self.d3, self.d4)
            if y1 < y2:
                b = c
            else:
                a = c
        optimal_x = (a + b) / 2
        df = pd.DataFrame(data)
        return optimal_x, df

    def golden_section_method(self):
        iterations = 0
        data = {'Ітерація k': [], 'ak': [], 'bk': [], 'xk(c)': [], '|bk-ak|': []}
        golden_ratio = (1 + np.sqrt(5)) / 2
        a, b = self.a, self.b
        x1 = b - (b - a) / golden_ratio
        x2 = a + (b - a) / golden_ratio
        while abs(b - a) >= self.epsilon:
            iterations += 1
            data['Ітерація k'].append(iterations)
            data['ak'].append(a)
            data['bk'].append(b)
            data['xk(c)'].append((a + b) / 2)
            data['|bk-ak|'].append(abs(b - a))
            y1 = self.f(x1, self.d1, self.d2, self.d3, self.d4)
            y2 = self.f(x2, self.d1, self.d2, self.d3, self.d4)
            if y1 < y2:
                b = x2
                x2 = x1
                x1 = b - (b - a) / golden_ratio
            else:
                a = x1
                x1 = x2
                x2 = a + (b - a) / golden_ratio
        optimal_x = (a + b) / 2
        df = pd.DataFrame(data)
        return optimal_x, df


# Визначення параметрів
def f(x, d1, d2, d3, d4):
    return d1 * x**3 + d2 * x**2 + d3 * x + d4

d1 = 1
d2 = -4
d3 = -20
d4 = 85
epsilon = 0.01
a = 0
b = 10

# Створення об'єкту для методів оптимізації
optimizer = OptimizationMethods(f, a, b, epsilon, d1, d2, d3, d4)

# Знаходимо мінімум методом половинного ділення
optimal_x_bisection, df_bisection = optimizer.bisection_method()
print("\nПошук оптимального рішення (метод половинного ділення):")
print(df_bisection)
print("Оптимальне значення х =", optimal_x_bisection)
print("Мінімальне значення функції f(x) =", f(optimal_x_bisection, d1, d2, d3, d4))

# Знаходимо мінімум методом золотого перерізу
optimal_x_golden_section, df_golden_section = optimizer.golden_section_method()
print("\nПошук оптимального рішення (метод золотого перерізу):")
print(df_golden_section)
print("Оптимальне значення х =", optimal_x_golden_section)
print("Мінімальне значення функції f(x) =", f(optimal_x_golden_section, d1, d2, d3, d4))

# Побудова графіка цільової функції
x_values = np.linspace(0, 10, 100)
y_values = f(x_values, d1, d2, d3, d4)
plt.plot(x_values, y_values, label='f(x)')
plt.scatter(optimal_x_bisection, f(optimal_x_bisection, d1, d2, d3, d4), color='red', label='Мінімум методом половинного ділення')
plt.scatter(optimal_x_golden_section, f(optimal_x_golden_section, d1, d2, d3, d4), color='green', label='Мінімум методом золотого перерізу')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Графік цільової функції та методи пошуку мінімуму')
plt.legend()
plt.grid(True)
plt.show()
