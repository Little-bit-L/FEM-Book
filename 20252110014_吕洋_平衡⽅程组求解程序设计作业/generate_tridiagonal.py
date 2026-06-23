import numpy as np
import json

def generate_tridiagonal(n, output_file):
    K = np.zeros((n, n))
    for i in range(n):
        K[i, i] = 2.0
        if i > 0:
            K[i, i-1] = -1.0
            K[i-1, i] = -1.0
    a_exact = np.ones(n)
    R = K @ a_exact
    data = {
        "Title": f"三对角矩阵 n={n}",
        "n": n,
        "K": K.tolist(),
        "R": R.tolist(),
        "exact_solution": a_exact.tolist()
    }
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Generated {output_file}")

if __name__ == "__main__":
    for n in [10, 100]:
        generate_tridiagonal(n, f"tridiagonal_n{n}.json")