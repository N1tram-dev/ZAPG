import time
import numpy as np
import matplotlib.pyplot as plt


def gauss_solve(A, b, pivot=True, return_intermediate=False, verbose=False):
    """
    Parametry patřebné na vložení (vpisujte do programu sekce test)
    ----------
    A : (n, n) matice určitě čtverec
    b : (n,) rozšíření

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

    # Forward elimination
    for k in range(n - 1):
        # Partial pivoting
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

        # Eliminate entries below the pivot
        for i in range(k + 1, n):
            m = U[i, k] / U[k, k]
            U[i, k:] -= m * U[k, k:]
            bb[i] -= m * bb[k]

        if verbose:
            print(f"po eliminaci sloupce {k}:\n{U}\n")

    # Back substitution
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        s = np.dot(U[i, i + 1:], x[i + 1:])
        x[i] = (bb[i] - s) / U[i, i]

    if return_intermediate:
        return x, U, bb
    return x


def well_conditioned_random_system(n, rng):
    """
    Funkce na tvorbu náhodného systému matice (A, b).
    n*E omezuje možnosti výskytu singulárních matic pro benchmark.
    """
    A = rng.standard_normal((n, n))
    A += n * np.eye(n)   # diagonal dominance helps stability
    b = rng.standard_normal(n)
    return A, b


def benchmark(sizes=(10, 100, 300, 500), trials=3, pivot=True):
    """
    Časové zhodnoceni fce gauss_solve vs np.linalg.solve skrz různé velikosti matic.
    Vrací: sizes_list, times_gauss, times_np
    """
    rng = np.random.default_rng(42)
    sizes_list = []
    times_gauss = []
    times_np = []

    for n in sizes:
        # average over 'trials' random systems
        t_g_sum = 0.0
        t_np_sum = 0.0
        for _ in range(trials):
            A, b = well_conditioned_random_system(n, rng)

            # time gauss_solve
            t0 = time.perf_counter()
            _ = gauss_solve(A, b, pivot=pivot, return_intermediate=False, verbose=False)
            t_g_sum += time.perf_counter() - t0

            # time np.linalg.solve
            t0 = time.perf_counter()
            _ = np.linalg.solve(A, b)
            t_np_sum += time.perf_counter() - t0

        sizes_list.append(n)
        times_gauss.append(t_g_sum / trials)
        times_np.append(t_np_sum / trials)

    return np.array(sizes_list), np.array(times_gauss), np.array(times_np)


def main():
    # --- Part 1: Solve the provided 3x3 example and print U, bb, x
    A = np.array([[ 2,  1, -1],
                  [-3, -1,  2],
                  [-2,  1,  2]], dtype=float)
    b = np.array([8, -11, -3], dtype=float)

    x, U, bb = gauss_solve(A, b, return_intermediate=True, verbose=False)
    print("Vrchní trojúhelníkový tvar U:\n", U)
    print("Změněná rozšířená matice b:\n", bb)
    print("Vektor x:\n", x)

    # --- Part 2: Benchmark and plot
    sizes, t_gauss, t_np = benchmark(sizes=(10, 100, 300, 500), trials=3, pivot=True)

    plt.figure()
    plt.plot(sizes, t_gauss, marker='o', label='Gaussian elimination')
    plt.plot(sizes, t_np, marker='o', label='np.linalg.solve')
    plt.xlabel('Maticová velikost n (n×n)')
    plt.ylabel('průměrný čas řešení (seconds)')
    plt.title('Čas řešení vs. maticová velikost')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
