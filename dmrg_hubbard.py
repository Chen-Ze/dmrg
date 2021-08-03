import numpy as np
from scipy.sparse.linalg import eigsh

from collections import namedtuple

HamiltonianParameters   = namedtuple("HamiltonianParameters", ["t", "U"])
Block                   = namedtuple(
    "Block",
    ["length", "dimension", "operators", "result", "up_creations", "down_creations"],
    defaults=(None, None, None, None)
)

model_d                 = 2 # number of states on each site of a fixed spin: occupied / not occupied
creation                = np.matrix([[0, 0], [1, 0]], dtype='d') # occupied as [0, 1]' and not occupied as [1, 0]'
annihilation            = creation.H
up_creation             = np.kron(creation, np.eye(model_d))
up_annihilation         = np.kron(annihilation, np.eye(model_d))
down_creation           = np.kron(np.eye(model_d), creation)
down_annihilation       = np.kron(np.eye(model_d), annihilation)

def initial_block(hamiltonianParameters):
    # hamiltonian = (
    #     hamiltonianParameters.U * (
    #         up_creation * up_annihilation * down_creation * down_annihilation
    #     )
    # )
    operators = {
        "hamiltonian":          np.matrix([[0]], dtype='d'),
        "border_up_creation":   np.matrix([[0]], dtype='d'),
        "border_down_creation": np.matrix([[0]], dtype='d')
    }
    return Block(0, 1, operators, None, [], [])
    

def enlarge_block(block, hamiltonianParameters):
    old_d = block.dimension
    old_hamiltonian                 = block.operators["hamiltonian"]
    old_border_up_creation          = block.operators["border_up_creation"]
    old_border_down_creation        = block.operators["border_down_creation"]

    new_site_hamiltonian = (
        -hamiltonianParameters.t * (
            np.kron(old_border_up_creation,   up_annihilation) +
            np.kron(old_border_down_creation, down_annihilation) +
            np.kron(old_border_up_creation,   up_annihilation).H +
            np.kron(old_border_down_creation, down_annihilation).H
        ) +
        hamiltonianParameters.U * (
            np.kron(
                np.eye(old_d),
                up_creation * up_annihilation * down_creation * down_annihilation
            )
        )
    )
    new_hamiltonian = (
        np.kron(old_hamiltonian, np.eye(model_d * model_d)) + # the old part
        new_site_hamiltonian
    )

    new_border_up_creation      = np.kron(np.eye(old_d), up_creation)
    new_border_down_creation    = np.kron(np.eye(old_d), down_creation)

    new_operators = {
        "hamiltonian":          new_hamiltonian,
        "border_up_creation":   new_border_up_creation,
        "border_down_creation": new_border_down_creation
    }

    new_upcreations   = list(map(lambda c: np.kron(c, np.eye(model_d * model_d)), block.up_creations))
    new_downcreations = list(map(lambda c: np.kron(c, np.eye(model_d * model_d)), block.down_creations))
    new_upcreations.append(new_border_up_creation)
    new_downcreations.append(new_border_down_creation)

    new_block = Block(block.length + 1, old_d * model_d * model_d, new_operators, None, new_upcreations, new_downcreations)
    return new_block

def conjugacy(operator, transformation):
    assert isinstance(operator,       np.matrix)
    assert isinstance(transformation, np.matrix)
    return transformation.H * operator * transformation
    
def single_dmrg_step(sys_block, env_block, hamiltonianParameters, m):
    """Keeping only `m` states in the new basis.
    """
    sys_block   = enlarge_block(sys_block, hamiltonianParameters)
    sys_d       = sys_block.dimension
    env_block   = enlarge_block(env_block, hamiltonianParameters)
    env_d       = env_block.dimension

    superblock_hamiltonian = (
        np.kron(sys_block.operators["hamiltonian"], np.eye(env_d)) +
        np.kron(np.eye(sys_d), env_block.operators["hamiltonian"]) +
        (-hamiltonianParameters.t) * (
            np.kron(
                sys_block.operators["border_up_creation"],
                env_block.operators["border_up_creation"].H
            ) +
            np.kron(
                sys_block.operators["border_up_creation"],
                env_block.operators["border_up_creation"].H
            ).H +
            np.kron(
                sys_block.operators["border_down_creation"],
                env_block.operators["border_down_creation"].H
            ) +
            np.kron(
                sys_block.operators["border_down_creation"],
                env_block.operators["border_down_creation"].H
            ).H
        )
    )
    (energy,), psi0 = eigsh(superblock_hamiltonian, k=1, which="SA") # find the lowest eigenvalue

    psi0 = psi0.reshape([sys_d, -1])
    rho  = np.dot(psi0, psi0.conjugate().transpose())

    up_densities   = list(map(lambda c: np.trace(c * c.H * np.matrix(rho)), sys_block.up_creations))
    down_densities = list(map(lambda c: np.trace(c * c.H * np.matrix(rho)), sys_block.down_creations))

    # Projection to the most significant `m` eigenvectors
    # ***** Code below copied from `simple_dmrg_02_finite_system.py` *****
    # Diagonalize the reduced density matrix and sort the eigenvectors by eigenvalue.
    evals, evecs = np.linalg.eigh(rho)
    possible_eigenstates = []
    for eval, evec in zip(evals, evecs.transpose()):
        possible_eigenstates.append((eval, evec))
    possible_eigenstates.sort(reverse=True, key=lambda x: x[0])  # largest eigenvalue first

    # Build the transformation matrix from the `m` overall most significant
    # eigenvectors.
    my_m = min(len(possible_eigenstates), m)
    transformation_matrix = np.zeros((sys_d, my_m), dtype='d', order='F')
    for i, (eval, evec) in enumerate(possible_eigenstates[:my_m]):
        transformation_matrix[:, i] = evec

    truncation_error = 1 - sum([x[0] for x in possible_eigenstates[:my_m]])
    print("truncation error:", truncation_error)

    transformation_matrix = np.matrix(transformation_matrix)

    # Rotate and truncate each operator.
    new_operators = {}
    for name, op in sys_block.operators.items():
        new_operators[name] = conjugacy(op, transformation_matrix)

    newblock = Block(sys_block.length, my_m, new_operators,
        { "up_densities": up_densities, "down_densities": down_densities },
        list(map(lambda c: conjugacy(c, transformation_matrix), sys_block.up_creations)),
        list(map(lambda c: conjugacy(c, transformation_matrix), sys_block.down_creations))
    )

    return newblock, energy

class DMRG:
    def __init__(self, hamiltonianParameters):
        self.hamiltonianParameters  = hamiltonianParameters
        self.sys_block              = initial_block(hamiltonianParameters)
        self.env_block              = initial_block(hamiltonianParameters)
        self.sys_label              = "l"
        self.env_label              = "r"
        self.block_disk             = {}
        self.save_blocks()

    def save_blocks(self):
        self.block_disk[self.sys_label, self.sys_block.length] = self.sys_block
        self.block_disk[self.env_label, self.env_block.length] = self.env_block

    def graphic(self):
        # ***** Code below copied from `simple_dmrg_02_finite_system.py` *****
        graphic = ("=" * self.sys_block.length) + "**" + ("-" * self.env_block.length)
        if self.sys_label == "r":
            graphic = graphic[::-1]
        return graphic

    def infinite_system_dmrg(self, l, m):
        print("***** Infinite System DMRG *****")
        while 2 * self.sys_block.length < l:
            print(f"L = {2 * self.sys_block.length + 2}")
            new_block, energy = single_dmrg_step(
                self.sys_block, self.env_block, self.hamiltonianParameters, m
            )
            self.sys_block = new_block
            self.env_block = new_block
            print(f"E/L = {energy / (2 * self.sys_block.length)}")
            self.save_blocks()

    def finite_system_dmrg(self, l, m_warmup, m_sweep_list):
        # ***** Code below copied from `simple_dmrg_02_finite_system.py` *****
        assert l % 2 == 0

        self.infinite_system_dmrg(l, m_warmup)
        
        print("***** Finite System DMRG *****")

        for m in m_sweep_list:
            while True:
                self.env_block = self.block_disk[self.env_label, l - self.sys_block.length - 2]
                if self.env_block.length == 0:
                    self.sys_block, self.env_block = self.env_block, self.sys_block
                    self.sys_label, self.env_label = self.env_label, self.sys_label
                
                print(self.graphic())
                self.sys_block, energy = single_dmrg_step(
                    self.sys_block, self.env_block, self.hamiltonianParameters, m
                )
                print(f"E/L = {energy / l}")

                self.block_disk[self.sys_label, self.sys_block.length] = self.sys_block

                if self.sys_label == "l" and 2 * self.sys_block.length == l:
                    break

        return self.block_disk["l", l / 2], self.block_disk["r", l / 2]


if __name__ == "__main__":
    np.set_printoptions(precision=10, suppress=True, threshold=10000, linewidth=300)

    dmrg = DMRG(HamiltonianParameters(t=0.5, U=0))
    l_block, r_block = dmrg.finite_system_dmrg(24, 25, [25])
    print(l_block.result)
    print(r_block.result)

