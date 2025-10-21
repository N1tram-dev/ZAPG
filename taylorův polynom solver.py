import numpy as np
import sympy as sp
import math
import matplotlib.pyplot as plt
import pandas as pd

# ---------- 1) Funkce a náš Taylor (Maclaurin) ----------
def f(x):
    """Cílová funkce."""
    return np.sin(x)

def taylor_sin_maclaurin(x, n):
    """
    Maclaurinův polynom sin(x) do stupně n (včetně).
    Přidává pouze liché mocniny: x, x^3/3!, x^5/5!, ...
    """
    x = np.array(x, dtype=float)
    s = np.zeros_like(x)
    # term s mocninou 2k+1, dokud 2k+1 <= n
    for k in range(0, n + 1):
        deg = 2*k + 1
        if deg > n:
            break
        s += ((-1)**k) * (x**deg) / math.factorial(deg)
    return s

# ---------- 2) Nastavení ----------
orders = [2, 3, 4, 6, 8]
x = np.linspace(-np.pi, np.pi, 2000)
y = f(x)

# ---------- 3) Graf: sin vs Taylor ----------
plt.figure()
plt.plot(x, y, label="sin(x)")
for n in orders:
    plt.plot(x, taylor_sin_maclaurin(x, n), label=f"Taylor n={n}")
plt.title("sin(x) vs. Maclaurinovy polynomy (n = 2,3,4,6,8) na [-π, π]")
plt.xlabel("x"); plt.ylabel("y"); plt.grid(True); plt.legend()
plt.show()

# ---------- 4) Graf: absolutní chyba ----------
plt.figure()
for n in orders:
    err = np.abs(y - taylor_sin_maclaurin(x, n))
    plt.plot(x, err, label=f"|error| n={n}")
plt.title("Absolutní chyba |sin(x) - T_n(x)|")
plt.xlabel("x"); plt.ylabel("absolutní chyba"); plt.grid(True); plt.legend()
plt.show()

# ---------- 5) Zoom kolem nuly ----------
x_zoom = np.linspace(-1.2, 1.2, 1200)
y_zoom = f(x_zoom)
plt.figure()
plt.plot(x_zoom, y_zoom, label="sin(x)")
for n in orders:
    plt.plot(x_zoom, taylor_sin_maclaurin(x_zoom, n), label=f"Taylor n={n}")
plt.title("Zoom u 0: sin(x) vs. Taylorovy polynomy")
plt.xlabel("x"); plt.ylabel("y"); plt.grid(True); plt.legend()
plt.show()

# ---------- 6) Porovnání se sympy.series() ----------
symx = sp.symbols('x')
comparison_rows = []
test_pts = np.array([-np.pi/2, -1.0, -0.5, 0.0, 0.5, 1.0, np.pi/2])

for n in orders:
    # sympy Taylor kolem 0 až do řádu n (O(x^{n+1}) je oříznuto)
    series_poly = sp.series(sp.sin(symx), symx, 0, n+1).removeO()
    series_fun = sp.lambdify(symx, series_poly, 'numpy')

    ours_vals = taylor_sin_maclaurin(test_pts, n)
    theirs_vals = series_fun(test_pts)
    max_diff = float(np.max(np.abs(ours_vals - theirs_vals)))

    comparison_rows.append({
        "n": n,
        "sympy_polynomial": str(sp.simplify(series_poly)),
        "max |our - sympy| on test pts": f"{max_diff:.3e}"
    })

comparison_df = pd.DataFrame(comparison_rows)
print("\n=== Porovnání s `sympy.series()` (symbolický Taylor) ===")
print(comparison_df.to_string(index=False))

# ---------- 7) Chybové statistiky ----------
def error_stats(xgrid, func, approx):
    err = np.abs(func - approx)
    return float(np.max(err)), float(np.sqrt(np.mean(err**2)))

rows = []
mask_small = (x >= -1) & (x <= 1)
for n in orders:
    approx_full = taylor_sin_maclaurin(x, n)
    max_full, rms_full = error_stats(x, y, approx_full)

    x_small = x[mask_small]
    y_small = y[mask_small]
    approx_small = taylor_sin_maclaurin(x_small, n)
    max_small, rms_small = error_stats(x_small, y_small, approx_small)

    rows.append({
        "n": n,
        "max error [-π,π]": f"{max_full:.6f}",
        "RMS error [-π,π]": f"{rms_full:.6f}",
        "max error [-1,1]": f"{max_small:.6f}",
        "RMS error [-1,1]": f"{rms_small:.6f}",
    })

errors_df = pd.DataFrame(rows)
print("\n=== Aproximační chyba (max & RMS) ===")
print(errors_df.to_string(index=False))

# ---------- 8) Poznámky k interpretaci ----------
# - Taylorova řada nejlépe aproximuje v okolí bodu rozvoje (zde a=0).
# - S rostoucím n klesá chyba hlavně blízko nuly; dál od 0 chyba roste.
# - Pro větší intervaly je výhodné zvýšit n nebo volit jiný bod rozvoje.
