import numpy as np
import util
import functools
import itertools

from numpy import eye as assymetrizer
# from util import sigma_product as assymetrizer

t = 0.5 # hopping t
U = 1.0 # on-site U
L = 6   # number of sites

spin_d                  = 2 # number of states on each site of a fixed spin: occupied / not occupied
site_d                  = spin_d * spin_d # number of states on each site
creation                = np.matrix([[0, 0], [1, 0]], dtype='d') # occupied as [0, 1]' and not occupied as [1, 0]'
up_creation             = np.kron(creation, np.eye(spin_d))
down_creation           = np.kron(util.sigma_product(spin_d), creation) # np.kron(util.sigma_product(model_d), creation)


up_creation_operators, down_creation_operators = map(
    lambda creation: [
        functools.reduce(np.kron,
            (util.sigma_product(site_d ** i), creation, np.eye(site_d ** (L - 1 - i)))
        )
        for i in range(L)
    ],
    (up_creation, down_creation)
)

up_hoppings   = map(lambda creation_0, creation_1: -t * (creation_0 * creation_1.H), up_creation_operators[:-1], up_creation_operators[1:])
down_hoppings = map(lambda creation_0, creation_1: -t * (creation_0 * creation_1.H), down_creation_operators[:-1], down_creation_operators[1:])
on_sites      = map(lambda creation_up, creation_down: U * creation_up * creation_up.H * creation_down * creation_down.H, up_creation_operators, down_creation_operators)

H = sum([(hopping + hopping.H) for hopping in itertools.chain(up_hoppings, down_hoppings)]) + sum(on_sites)

assert np.allclose(H, H.H) # assert H is Hermitian
print(min(np.linalg.eigh(H)[0]) / L)
