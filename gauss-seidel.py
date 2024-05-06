import numpy as np

def print_equations(A, B):
    n = len(B)
    for i in range(n):
        equation = f"{A[i, 0]}x1"
        for j in range(1, n):
            equation += f" + {A[i, j]}x{j + 1}"
        equation += f" = {B[i]}"
        print(equation)

def check_diagonal_domination(A):
    diagonal_sum = np.abs(np.diag(A)).sum()
    off_diagonal_sum = (np.abs(A) - np.abs(np.diag(A))).sum()

    return diagonal_sum >= off_diagonal_sum

def diagonalize_matrix(A):
    if check_diagonal_domination(A):
        return A, None, None

    eigenvalues, eigenvectors = np.linalg.eig(A)
    if np.linalg.matrix_rank(eigenvectors) == A.shape[0]:
        P = eigenvectors
        D = np.diag(eigenvalues)
        P_inv = np.linalg.inv(P)
        diagonalized_A = P @ D @ P_inv
        return diagonalized_A, P, D
    else:
        print("Неможливо діагоналізувати: неповний набір власних векторів.")

def gauss_seidel(A, B, tolerance=1e-3, max_iterations=10):
    n = len(B)
    X = np.zeros(n)

    print("System of Equations:")
    print_equations(A, B)
    
    for iteration in range(max_iterations+1):
        print(f'ITERATION {iteration}')
        X_old = np.copy(X)
        
        for i in range(n):
            sigma = np.dot(A[i, :i], X[:i]) + np.dot(A[i, i+1:], X_old[i+1:])
            X[i] = (B[i] - sigma) / A[i, i]
        
        print(X)
        print(f"Error: {np.linalg.norm(X - X_old, ord=np.inf)}")
            
        if np.linalg.norm(X - X_old, ord=np.inf) < tolerance:
            break
    
    return X

A = np.array([[12, 0, 0, 3],
            [2, 20, -3, 0],
            [0, 2, 16, -1],
            [4, -2, 0, 24],])

B = np.array([18, 39, -25, 0])

diagonalized_A, P, D = diagonalize_matrix(A)
solution = gauss_seidel(diagonalized_A, B)

print("Solution:", solution)


print("EXACT Solution:", np.linalg.solve(A, B))