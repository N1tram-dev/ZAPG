import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu_factor, lu_solve  # Nová závislost pro LU rozklad


def gauss_solve(A, b, pivot=True, return_intermediate=False, verbose=False):
    """
    Parametry patřebné na vložení (vpisujte do programu sekce test)
    ----------
    A : (n, n) matice určitě čtverec
    b : (n, 1) rozšíření (pravá strana)

    Hodnoty které program vrátí
    -------
    x vektor, U upravená matice, bb upracená pravá strana
    """
    A = np.array(A, dtype=float, copy=True)
    b = np.array(b, dtype=float, copy=True)

    n, m = A.shape
    if n != m:
        raise ValueError("Matice musí být čtvercová.")
    if b.size != n:
        raise ValueError("b musí odpovídat počtu neznámých n-tého řádku.")

    U = A.copy()
    bb = b.copy()

    # eliminace od předu
    for k in range(n - 1):
        # parciální pivot
        if pivot:
            i_max = k + np.argmax(np.abs(U[k:, k]))
            if np.isclose(U[i_max, k], 0.0):
                raise np.linalg.LinAlgError("Matice je singular.")
            if i_max != k:
                U[[k, i_max], :] = U[[i_max, k], :]
                bb[[k, i_max]] = bb[[i_max, k]]
        else:
            if np.isclose(U[k, k], 0.0):
                raise np.linalg.LinAlgError("pivot je nula; enable pivoting.")

        # Pivotní bod eliminace
        for i in range(k + 1, n):
            m = U[i, k] / U[k, k]
            U[i, k:] -= m * U[k, k:]
            bb[i] -= m * bb[k]

        if verbose:
            print(f"po eliminaci sloupce {k}:\n{U}\n")

    # Zpětná substituce
    x = np.zeros(n, dtype=float)
    # Kontrola nenulového prvku na diagonále
    if np.isclose(U[n - 1, n - 1], 0.0):
        raise np.linalg.LinAlgError("Matice je singular při zpětné substituci.")

    for i in range(n - 1, -1, -1):
        # Dvojitá kontrola, i když by měla být detekována pivotací/singularitou
        if np.isclose(U[i, i], 0.0):
            raise np.linalg.LinAlgError("Nulový pivot při zpětné substituci.")
        s = np.dot(U[i, i + 1:], x[i + 1:])
        x[i] = (bb[i] - s) / U[i, i]

    if return_intermediate:
        return x, U, bb
    return x


# --- NOVÁ FUNKCE: Řešení pomocí LU rozkladu ---
def lu_solve_system(A, b):
    """
    Řešení soustavy Ax=b pomocí LU rozkladu z knihovny SciPy.
    """
    A_copy = np.array(A, dtype=float, copy=True)
    b_copy = np.array(b, dtype=float, copy=True)

    # LU rozklad A = P*L*U
    lu, piv = lu_factor(A_copy)

    # Řešení soustavy P*L*U*x = b, tj. L*U*x = P^T*b
    x = lu_solve((lu, piv), b_copy)
    return x


# --- NOVÁ FUNKCE: Výpočet normy chyby (rezidua) ---
def calculate_error_norm(A, b, x):
    """
    Vypočítá normu chyby rezidua: ||Ax - b||_2
    """
    return np.linalg.norm(np.dot(A, x) - b, ord=2)


def well_conditioned_random_system(n, rng):
    """
    Funkce na tvorbu náhodného systému matice (A, b).
    n*E omezuje možnosti výskytu singulárních matic pro benchmark.
    """
    A = rng.standard_normal((n, n))
    A += n * np.eye(n)  # diagonal dominance helps stability
    # Vytvoření přesného řešení, pro které se pak b spočítá
    x_true = rng.standard_normal(n)
    b = np.dot(A, x_true)
    return A, b, x_true  # Vrací i přesné řešení x_true


def benchmark(sizes=(10, 100, 300, 500), trials=3, pivot=True):
    """
    Časové zhodnoceni fce gauss_solve vs np.linalg.solve vs lu_solve skrz různé velikosti matic.
    Vrací: sizes_list, times_gauss, times_np, times_lu, error_gauss, error_lu
    """
    rng = np.random.default_rng(42)
    sizes_list = []
    times_gauss = []
    times_np = []
    times_lu = []  # Nový seznam pro časy LU
    error_gauss = []  # Nový seznam pro chyby
    error_lu = []

    for n in sizes:
        # average over 'trials' random systems
        t_g_sum = 0.0
        t_np_sum = 0.0
        t_lu_sum = 0.0
        err_g_sum = 0.0
        err_lu_sum = 0.0

        for _ in range(trials):
            A, b, x_true = well_conditioned_random_system(n, rng)

            # time gauss_solve
            t0 = time.perf_counter()
            x_gauss = gauss_solve(A, b, pivot=pivot, return_intermediate=False, verbose=False)
            t_g_sum += time.perf_counter() - t0
            err_g_sum += calculate_error_norm(A, b, x_gauss)

            # time np.linalg.solve
            t0 = time.perf_counter()
            _ = np.linalg.solve(A, b)
            t_np_sum += time.perf_counter() - t0

            # time lu_solve_system
            t0 = time.perf_counter()
            x_lu = lu_solve_system(A, b)
            t_lu_sum += time.perf_counter() - t0
            err_lu_sum += calculate_error_norm(A, b, x_lu)

        sizes_list.append(n)
        times_gauss.append(t_g_sum / trials)
        times_np.append(t_np_sum / trials)
        times_lu.append(t_lu_sum / trials)  # Průměrný čas LU
        error_gauss.append(err_g_sum / trials)  # Průměrná chyba Gaussovy eliminace
        error_lu.append(err_lu_sum / trials)  # Průměrná chyba LU rozkladu

    return np.array(sizes_list), np.array(times_gauss), np.array(times_np), np.array(times_lu), np.array(
        error_gauss), np.array(error_lu)


def main():
    # --- ZDE je TEST pro 3x3 výsledky jsou U, b, x
    A = np.array([[2, 1, -1],
                  [-3, -1, 2],
                  [-2, 1, 2]], dtype=float)
    b = np.array([8, -11, -3], dtype=float)

    print("--- Test pro 3x3 soustavu (Gaussova eliminace) ---")
    x, U, bb = gauss_solve(A, b, return_intermediate=True, verbose=False)
    print("Vrchní trojúhelníkový tvar U:\n", U)
    print("Změněná rozšířená matice b:\n", bb)
    print("Vektor x (Gaussova eliminace):\n", x)
    # Kontrola řešení přes np.linalg.solve
    x_np = np.linalg.solve(A, b)
    print("Vektor x (np.linalg.solve):\n", x_np)
    # Kontrola řešení přes LU
    x_lu = lu_solve_system(A, b)
    print("Vektor x (LU rozklad):\n", x_lu)

    # Výpočet chyby
    err_g = calculate_error_norm(A, b, x)
    err_lu = calculate_error_norm(A, b, x_lu)
    print(f"\nNorma chyby ||Ax - b||_2 (Gaussova eliminace): {err_g:.2e}")
    print(f"Norma chyby ||Ax - b||_2 (LU rozklad): {err_lu:.2e}")

    # --- Část 2: Benchmark a plot ---
    print("\n--- Benchmark a vykreslení ---")
    sizes, t_gauss, t_np, t_lu, err_gauss, err_lu = benchmark(
        sizes=(10, 100, 300, 500),  # Přidána jedna velikost pro lepší křivku
        trials=3,
        pivot=True
    )

    # --- Plot 1: Srovnání časů ---
    plt.figure(figsize=(10, 5))
    plt.plot(sizes, t_gauss, marker='o', label='Gaussova eliminace (vlastní)')
    plt.plot(sizes, t_lu, marker='o', label='LU rozklad (scipy)')  # Vykreslení LU
    plt.plot(sizes, t_np, marker='o', label='np.linalg.solve')
    plt.xlabel('Maticová velikost n (n×n)')
    plt.ylabel('průměrný čas řešení (seconds)')
    plt.title('Čas řešení vs. maticová velikost')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #
    # --- Plot 2: Srovnání chyby řešení ---
    plt.figure(figsize=(10, 5))
    plt.plot(sizes, err_gauss, marker='o', label='Gaussova eliminace - chyba ||Ax-b||_2', linestyle='--')
    plt.plot(sizes, err_lu, marker='s', label='LU rozklad (scipy) - chyba ||Ax-b||_2', linestyle='--')
    plt.xlabel('Maticová velikost n (n×n)')
    plt.ylabel('Norma chyby (Reziduum) ||Ax - b||_2')
    plt.title('Norma chyby řešení vs. maticová velikost')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Logaritmická stupnice pro lepší zobrazení malých chyb
    plt.tight_layout()
    #
    plt.show()


if __name__ == "__main__":
    main()