import numpy as np

sigma_z = np.matrix([[1, 0], [0, -1]], dtype='d')


def is_power_of_two(dimension):
    return (dimension & (dimension - 1) == 0) and dimension != 0

def sigma_product(dimension):
    dimension = int(dimension)
    assert is_power_of_two(dimension)
    return np.matrix([[1]], dtype='d') if dimension == 1 \
        else np.kron(sigma_z, sigma_product(dimension / 2))

if __name__ == "__main__":
    print(sigma_product(1))
    print(sigma_product(2))
    print(sigma_product(4))
    print(sigma_product(8))