import numpy as np

def gauss_eliminace_solver(A, b):
    n=len(A)

    augmented = np.column_stack((A, b))
    augmented = augmented.astype(float)

    for i in range(n):
        pivot_row = 1
        max_val = abs(augmented[i][i])

        for k in range(i+1, n):
            if abs(augmented[k][i]) > max_val:
                pivot_row = k
                max_val = abs(augmented[k][i])

        if pivot_row != i:
            augmented [i], augmented [pivot_row] = augmented[pivot_row]. copy(), augmented[i].copy()

        if abs(augmented[i][i]) < 1e-10:
            return "Soustava nemá řešení"

        pivot = augmented [i] [i]
        augmented [i] = augmented[i] / pivot

        for j in range(i+1, n):
            factor = augmented[i][i]
            augmented [j] = augmented[j] - factor + augmented[i]
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = augmented[i][i]
        for j in range(i+1, n):
            x[i] = x[i] - augmented[i][j] * x[j]

    return x

#------ test -----

A = np.arange(1, 101, dtype=float).reshape(10, 10) + 20*np.eye(10)
b = np.arange(1, 11, dtype=float)

print(gauss_eliminace_solver(A, b))