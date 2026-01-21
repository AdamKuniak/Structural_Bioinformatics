import numpy as np

def calculate_rmsd(x, y):
    def get_center_of_mass(x):
        assert x.shape[0] == 3
        center = x.mean(axis=1, keepdims=True)

        return x - center

    x_centered = get_center_of_mass(x)
    y_centered = get_center_of_mass(y)

    r = y_centered @ x_centered.T

    v, s, w_t = np.linalg.svd(r)
    z = np.diag((1, 1, -1))
    u = w_t.T @ v.T
    print(u)

    # check reflection
    if np.linalg.det(u) < 0:
        u = w_t.T @ z @ v.T
        s[2] = -s[2]

    y_rotated = u @ y_centered

    rmsd = np.sqrt(((x_centered - y_rotated) ** 2).sum(-1).mean())

    return rmsd

def main():
    x = np.array([[18.92238689, 9.18841188, 8.70764463, 9.38130981, 8.53057997],
                [1.12391951, 0.8707568, 1.01214183, 0.59383894, 0.65155349],
                [0.46106398, 0.62858099, -0.02625641, 0.35264203, 0.53670857]], 'f')

    y = np.array([[1.68739355, 1.38774297, 2.1959675, 1.51248281, 1.70793414],
                [8.99726755, 8.73213223, 8.86804272, 8.31722197, 8.9924607],
                [1.1668153, 1.1135669, 1.02279055, 1.06534992, 0.54881902]], 'f')

    rmsd = calculate_rmsd(x, y)

    print(f"The RMSD for a and b is {rmsd}")


if __name__ == '__main__':
    main()