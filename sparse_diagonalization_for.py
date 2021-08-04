from scipy import sparse
from scipy.sparse import linalg
import functools

sigma_z = sparse.bsr_matrix([[1, 0], [0, -1]])

def sigma_product(dimension):
    dimension = int(dimension)
    return (
        sparse.bsr_matrix([[1]])
    ) if dimension == 1 else (
        sparse.kron(sigma_z, sigma_product(dimension / 2))
    )


t = 0.5 # hopping t
U = 1.0 # on-site U
L = 10  # number of sites

d_spin          = 2 # number of states on each site of a fixed spin: occupied / not occupied
d_site          = d_spin * d_spin # number of states on each site
creation        = sparse.bsr_matrix([[0, 0], [1, 0]], dtype='d') # occupied as [0, 1]' and not occupied as [1, 0]'
up_creation     = sparse.kron(creation, sparse.eye(d_spin))
down_creation   = sparse.kron(sigma_product(d_spin), creation) # np.kron(util.sigma_product(model_d), creation)


up_creation_operators   = [None] * L
down_creation_operators = [None] * L

for i in range(L):
    print(f"Creating up_creation_operators[{i}] of {L}.")
    up_creation_operators[i] = functools.reduce(sparse.kron,
        (sigma_product(d_site ** i), up_creation, sparse.eye(d_site ** (L - 1 - i)))
    )
    print(f"Creating down_creation_operators[{i}] of {L}.")
    down_creation_operators[i] = functools.reduce(sparse.kron,
        (sigma_product(d_site ** i), down_creation, sparse.eye(d_site ** (L - 1 - i)))
    )

up_hoppings   = [None] * (L - 1)
down_hoppings = [None] * (L - 1)

for i in range(L - 1):
    print(f"Creating up_hoppings[{i}] of {L - 1}.")
    up_hoppings[i] = -t * (
        (up_creation_operators[i] * up_creation_operators[i + 1] .H) + 
        (up_creation_operators[i] * up_creation_operators[i + 1].H).H
    )
    print(f"Creating down_hoppings[{i}] of {L - 1}.")
    down_hoppings[i] = -t * (
        (down_creation_operators[i] * down_creation_operators[i + 1] .H) + 
        (down_creation_operators[i] * down_creation_operators[i + 1].H).H
    )

on_sites = [None] * L

for i in range(L):
    print(f"Creating on_sites[{i}] of {L}.")
    on_sites[i] = U * (
        up_creation_operators[i] * up_creation_operators[i].H *
        down_creation_operators[i] * down_creation_operators[i].H
    )

print("Summing operators.")
H = sum(up_hoppings) + sum(down_hoppings) + sum(on_sites)

assert H.shape == (d_site ** L, d_site ** L)
print(linalg.eigsh(H, k=1, which="SA")[0][0] / L)