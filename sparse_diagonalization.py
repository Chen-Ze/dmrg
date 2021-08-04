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
L = 4   # number of sites

d_spin          = 2 # number of states on each site of a fixed spin: occupied / not occupied
d_site          = d_spin * d_spin # number of states on each site
creation        = sparse.bsr_matrix([[0, 0], [1, 0]], dtype='d') # occupied as [0, 1]' and not occupied as [1, 0]'
up_creation     = sparse.kron(creation, sparse.eye(d_spin))
down_creation   = sparse.kron(sigma_product(d_spin), creation) # np.kron(util.sigma_product(model_d), creation)


up_creation_operators, down_creation_operators = ([
        functools.reduce(sparse.kron,
            (sigma_product(d_site ** i), creation, sparse.eye(d_site ** (L - 1 - i)))
        )
        for i in range(L)
    ] for creation in (up_creation, down_creation)
)

up_hoppings, down_hoppings = (map(
        lambda creation_0, creation_1: (
            -t * ((creation_0 * creation_1.H) + (creation_0 * creation_1.H).H)
        ),
        operators[:-1], operators[1:]
    ) for operators in (up_creation_operators, down_creation_operators)
)

on_sites = map(
    lambda creation_up, creation_down: (
        U * creation_up * creation_up.H * creation_down * creation_down.H
    ),
    up_creation_operators, down_creation_operators
)

H = sum(up_hoppings) + sum(down_hoppings) + sum(on_sites)

assert H.shape == (d_site ** L, d_site ** L)
print(linalg.eigsh(H, k=1, which="SA")[0][0] / L)