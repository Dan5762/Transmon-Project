import numpy as np
import scipy as sp
import scipy.sparse.linalg
import scipy.sparse as sparse
import matplotlib.pyplot as plt


class RCCircuit:
    def __init__(self, Ejp=121e9, Cjp=8e-15, Ccp=5e-15, alphas=[0.5, 0.5, 1, 1], ng=0.25, flux=0.5,
                 Cr=100e-15, Lr=10e-9, Ejt=5e9, Cjt=4e-15, Cct=5e-15, Ct=40e-15, ncut=2, mcut=20):
        self.h = 6.626e-34
        self.hbar = 1.055e-34
        self.e_charge = 1.60218e-19

        self.Ejp = Ejp * self.h                 # Probe qubit Josephson energy
        self.Cjp = Cjp                          # Probe qubit Josephson capacitance
        self.Ccp = Ccp                          # Probe qubit coupling capacitance
        self.alphas = alphas                    # Asymmetry of probe qubit
        self.ng = ng                            # Reduced gate charge
        self.flux = flux                        # Flux through probe qubit
        self.Cr = Cr                            # Resonator capacitance
        self.Lr = Lr                            # Resonator inductance
        self.Ejt = Ejt * self.h                 # Target qubit Josephson energy
        self.Cjt = Cjt                          # Target qubit Josephson capacitance
        self.Cct = Cct                          # Target qubit coupling capacitance
        self.Ct = Cct                           # Target qubit capacitance
        self.Z0 = np.sqrt(self.Lr / self.Cr)    # Impedance of resonator
        self.ncut = ncut                        # Cut-off threshold for number basis
        self.mcut = mcut                        # Cut-off threshold for resonator Fock states

        self.init_operators()

    def _print_params(self):
        print(f'Ejp:    {self.Ejp}')
        print(f'Cjp:    {self.Cjp}')
        print(f'Ccp:    {self.Ccp}')
        print(f'alpha: {self.alphas[0]}')
        print(f'ng:    {self.ng}')
        print(f'flux:  {self.flux}')
        print(f'Cr:    {self.Cr}')
        print(f'Lr:    {self.Lr}')
        print(f'Ejt:   {self.Ejt}')
        print(f'Cjt:   {self.Cjt}')
        print(f'Cct:   {self.Cct}')

    def init_operators(self):
        self.I_cb = sp.sparse.diags(np.ones(2 * self.ncut + 1, dtype=np.clongdouble))   # Identity for qubit (charge basis)
        self.I_fb = sp.sparse.diags(np.ones(self.mcut, dtype=np.clongdouble))           # Identity for resonator (Fock basis)

        self.q_op_cb = sp.sparse.diags(-2 * self.e_charge * np.arange(-self.ncut, self.ncut + 1, dtype=np.clongdouble))           # Charge operator (charge basis)
        self.ng_op_cb = -2 * self.e_charge * self.ng * self.I_cb
        self.e_iphi_op_cb = sp.sparse.diags(np.ones(2 * self.ncut, dtype=np.clongdouble), offsets=1)                                   # e^{i \phi} operator (charge basis)
        self.creation_op_fb = sp.sparse.diags(np.sqrt(np.arange(1, self.mcut, dtype=np.clongdouble)), offsets=-1)                     # Creation operator (Fock basis)
        self.annihilation_op_fb = np.conjugate(self.creation_op_fb.T)                                                 # Annihilation operator (Fock basis)
        self.q_op_fb = np.sqrt(self.hbar / (2 * self.Z0)) * 1j * (self.creation_op_fb - self.annihilation_op_fb)      # Charge operator (Fock basis)
        self.phi_op_fb = np.sqrt(self.hbar * self.Z0 * 0.5) * (self.creation_op_fb + self.annihilation_op_fb)         # Phi operator (Fock basis)

        self.q1_q1_prt = self.tensor5(self.q_op_cb @ self.q_op_cb, self.I_cb, self.I_cb, self.I_fb, self.I_cb)
        self.q1_q2_prt = self.tensor5(self.q_op_cb, self.q_op_cb, self.I_cb, self.I_fb, self.I_cb)
        self.q1_q3_prt = self.tensor5(self.q_op_cb, self.I_cb, self.q_op_cb + self.ng_op_cb, self.I_fb, self.I_cb)
        self.q1_q4_prt = self.tensor5(self.q_op_cb, self.I_cb, self.I_cb, self.q_op_fb, self.I_cb)
        self.q1_q5_prt = self.tensor5(self.q_op_cb, self.I_cb, self.I_cb, self.I_fb, self.q_op_cb)
        self.q2_q2_prt = self.tensor5(self.I_cb, self.q_op_cb @ self.q_op_cb, self.I_cb, self.I_fb, self.I_cb)
        self.q2_q3_prt = self.tensor5(self.I_cb, self.q_op_cb, self.q_op_cb + self.ng_op_cb, self.I_fb, self.I_cb)
        self.q2_q4_prt = self.tensor5(self.I_cb, self.q_op_cb, self.I_cb, self.q_op_fb, self.I_cb)
        self.q2_q5_prt = self.tensor5(self.I_cb, self.q_op_cb, self.I_cb, self.I_fb, self.q_op_cb)
        self.q3_q3_prt = self.tensor5(self.I_cb, self.I_cb, (self.q_op_cb + self.ng_op_cb) @ (self.q_op_cb + self.ng_op_cb), self.I_fb, self.I_cb)
        self.q3_q4_prt = self.tensor5(self.I_cb, self.I_cb, self.q_op_cb + self.ng_op_cb, self.q_op_fb, self.I_cb)
        self.q3_q5_prt = self.tensor5(self.I_cb, self.I_cb, self.q_op_cb + self.ng_op_cb, self.I_fb, self.q_op_cb)
        self.q4_q4_prt = self.tensor5(self.I_cb, self.I_cb, self.I_cb, self.q_op_fb @ self.q_op_fb, self.I_cb)
        self.q4_q5_prt = self.tensor5(self.I_cb, self.I_cb, self.I_cb, self.q_op_fb, self.q_op_cb)
        self.q5_q5_prt = self.tensor5(self.I_cb, self.I_cb, self.I_cb, self.I_fb, self.q_op_cb @ self.q_op_cb)

        self.q1_q1_pr = self.tensor4(self.q_op_cb @ self.q_op_cb, self.I_cb, self.I_cb, self.I_fb)
        self.q1_q2_pr = self.tensor4(self.q_op_cb, self.q_op_cb, self.I_cb, self.I_fb)
        self.q1_q3_pr = self.tensor4(self.q_op_cb, self.I_cb, self.q_op_cb + self.ng_op_cb, self.I_fb)
        self.q1_q4_pr = self.tensor4(self.q_op_cb, self.I_cb, self.I_cb, self.q_op_fb)
        self.q2_q2_pr = self.tensor4(self.I_cb, self.q_op_cb @ self.q_op_cb, self.I_cb, self.I_fb)
        self.q2_q3_pr = self.tensor4(self.I_cb, self.q_op_cb, self.q_op_cb + self.ng_op_cb, self.I_fb)
        self.q2_q4_pr = self.tensor4(self.I_cb, self.q_op_cb, self.I_cb, self.q_op_fb)
        self.q3_q3_pr = self.tensor4(self.I_cb, self.I_cb, (self.q_op_cb + self.ng_op_cb) @ (self.q_op_cb + self.ng_op_cb), self.I_fb)
        self.q3_q4_pr = self.tensor4(self.I_cb, self.I_cb, self.q_op_cb + self.ng_op_cb, self.q_op_fb)
        self.q4_q4_pr = self.tensor4(self.I_cb, self.I_cb, self.I_cb, self.q_op_fb @ self.q_op_fb)

        self.q1_p = self.tensor3(self.q_op_cb, self.I_cb, self.I_cb)
        self.q2_p = self.tensor3(self.I_cb, self.q_op_cb, self.I_cb)
        self.q3_p = self.tensor3(self.I_cb, self.I_cb, self.q_op_cb + self.ng_op_cb)

        self.q1_q1_p = self.tensor3(self.q_op_cb @ self.q_op_cb, self.I_cb, self.I_cb)
        self.q1_q2_p = self.tensor3(self.q_op_cb, self.q_op_cb, self.I_cb)
        self.q1_q3_p = self.tensor3(self.q_op_cb, self.I_cb, self.q_op_cb + self.ng_op_cb)
        self.q2_q2_p = self.tensor3(self.I_cb, self.q_op_cb @ self.q_op_cb, self.I_cb)
        self.q2_q3_p = self.tensor3(self.I_cb, self.q_op_cb, self.q_op_cb + self.ng_op_cb)
        self.q3_q3_p = self.tensor3(self.I_cb, self.I_cb, (self.q_op_cb + self.ng_op_cb) @ (self.q_op_cb + self.ng_op_cb))

        self.q5_t = self.q_op_cb

        self.q5_q5_t = self.q_op_cb @ self.q_op_cb

    def tensor3(self, op1, op2, op3):
        return sparse.kron(sparse.kron(op1, op2), op3)

    def tensor4(self, op1, op2, op3, op4):
        return sparse.kron(sparse.kron(sparse.kron(op1, op2), op3), op4)

    def tensor5(self, op1, op2, op3, op4, op5):
        return sparse.kron(sparse.kron(sparse.kron(sparse.kron(op1, op2), op3), op4), op5)

    def hc(self, state):
        return np.conjugate(state).T

    def kin_prt(self):
        self.init_operators()

        C_mat = [
            [(self.alphas[2] + self.alphas[0]) * self.Cjp, 0, -self.alphas[0] * self.Cjp, 0, 0],
            [0, (self.alphas[3] + self.alphas[1]) * self.Cjp, -self.alphas[1] * self.Cjp, 0, 0],
            [-self.alphas[0] * self.Cjp, -self.alphas[1] * self.Cjp, (self.alphas[0] + self.alphas[1]) * self.Cjp + self.Ccp, -self.Ccp, 0],
            [0, 0, -self.Ccp, self.Cr + self.Ccp + self.Cct, -self.Cct],
            [0, 0, 0, -self.Cct, self.Cjt + self.Cct + self.Ct]
        ]

        C_mat_in = np.linalg.inv(C_mat)

        kin_prt = C_mat_in[0][0] * self.q1_q1_prt
        kin_prt += C_mat_in[1][1] * self.q2_q2_prt
        kin_prt += C_mat_in[2][2] * self.q3_q3_prt
        kin_prt += C_mat_in[3][3] * self.q4_q4_prt
        kin_prt += C_mat_in[4][4] * self.q5_q5_prt
        kin_prt += 2 * C_mat_in[0][1] * self.q1_q2_prt
        kin_prt += 2 * C_mat_in[0][2] * self.q1_q3_prt
        kin_prt += 2 * C_mat_in[0][3] * self.q1_q4_prt
        kin_prt += 2 * C_mat_in[0][4] * self.q1_q5_prt
        kin_prt += 2 * C_mat_in[1][2] * self.q2_q3_prt
        kin_prt += 2 * C_mat_in[1][3] * self.q2_q4_prt
        kin_prt += 2 * C_mat_in[1][4] * self.q2_q5_prt
        kin_prt += 2 * C_mat_in[2][3] * self.q3_q4_prt
        kin_prt += 2 * C_mat_in[2][4] * self.q3_q5_prt
        kin_prt += 2 * C_mat_in[3][4] * self.q4_q5_prt

        kin_prt *= 0.5

        return kin_prt

    def kin_pr(self):
        self.init_operators()

        C_mat = [
            [(self.alphas[2] + self.alphas[0]) * self.Cjp, 0, -self.alphas[0] * self.Cjp, 0],
            [0, (self.alphas[3] + self.alphas[1]) * self.Cjp, -self.alphas[1] * self.Cjp, 0],
            [-self.alphas[0] * self.Cjp, -self.alphas[1] * self.Cjp, (self.alphas[0] + self.alphas[1]) * self.Cjp + self.Ccp, -self.Ccp],
            [0, 0, -self.Ccp, self.Cr + self.Ccp + self.Cct]
        ]

        C_mat_in = np.linalg.inv(C_mat)

        kin_prt = C_mat_in[0][0] * self.q1_q1_pr
        kin_prt += C_mat_in[1][1] * self.q2_q2_pr
        kin_prt += C_mat_in[2][2] * self.q3_q3_pr
        kin_prt += C_mat_in[3][3] * self.q4_q4_pr
        kin_prt += 2 * C_mat_in[0][1] * self.q1_q2_pr
        kin_prt += 2 * C_mat_in[0][2] * self.q1_q3_pr
        kin_prt += 2 * C_mat_in[0][3] * self.q1_q4_pr
        kin_prt += 2 * C_mat_in[1][2] * self.q2_q3_pr
        kin_prt += 2 * C_mat_in[1][3] * self.q2_q4_pr
        kin_prt += 2 * C_mat_in[2][3] * self.q3_q4_pr

        kin_prt *= 0.5

        return kin_prt

    def pot_prt(self):
        self.init_operators()

        pot_prt = self.Ejp * sum(self.alphas) * self.tensor5(self.I_cb, self.I_cb, self.I_cb, self.I_fb, self.I_cb)
        pot_prt += -self.Ejp * 0.5 * self.alphas[2] * self.tensor5(self.e_iphi_op_cb + self.e_iphi_op_cb.T, self.I_cb, self.I_cb, self.I_fb, self.I_cb)
        pot_prt += -self.Ejp * 0.5 * self.alphas[0] * self.tensor5(self.e_iphi_op_cb.T, self.I_cb, self.e_iphi_op_cb, self.I_fb, self.I_cb)
        pot_prt += -self.Ejp * 0.5 * self.alphas[0] * self.tensor5(self.e_iphi_op_cb, self.I_cb, self.e_iphi_op_cb.T, self.I_fb, self.I_cb)
        pot_prt += -self.Ejp * 0.5 * self.alphas[1] * np.exp(2j * np.pi * self.flux) * self.tensor5(self.I_cb, self.e_iphi_op_cb, self.e_iphi_op_cb.T, self.I_fb, self.I_cb)
        pot_prt += -self.Ejp * 0.5 * self.alphas[1] * np.exp(-2j * np.pi * self.flux) * self.tensor5(self.I_cb, self.e_iphi_op_cb.T, self.e_iphi_op_cb, self.I_fb, self.I_cb)
        pot_prt += -self.Ejp * 0.5 * self.alphas[3] * self.tensor5(self.I_cb, self.e_iphi_op_cb + self.e_iphi_op_cb.T, self.I_cb, self.I_fb, self.I_cb)
        pot_prt += self.Ejt * self.tensor5(self.I_cb, self.I_cb, self.I_cb, self.I_fb, self.I_cb)
        pot_prt += -self.Ejt * 0.5 * self.tensor5(self.I_cb, self.I_cb, self.I_cb, self.I_fb, self.e_iphi_op_cb + self.e_iphi_op_cb.T)
        pot_prt += 0.5 * self.Lr**-1 * self.tensor5(self.I_cb, self.I_cb, self.I_cb, self.phi_op_fb @ self.phi_op_fb, self.I_cb)

        return pot_prt

    def pot_pr(self):
        self.init_operators()

        pot_pr = self.Ejp * sum(self.alphas) * self.tensor4(self.I_cb, self.I_cb, self.I_cb, self.I_fb)
        pot_pr += -self.Ejp * 0.5 * self.alphas[2] * self.tensor4(self.e_iphi_op_cb + self.e_iphi_op_cb.T, self.I_cb, self.I_cb, self.I_fb)
        pot_pr += -self.Ejp * 0.5 * self.alphas[0] * self.tensor4(self.e_iphi_op_cb.T, self.I_cb, self.e_iphi_op_cb, self.I_fb)
        pot_pr += -self.Ejp * 0.5 * self.alphas[0] * self.tensor4(self.e_iphi_op_cb, self.I_cb, self.e_iphi_op_cb.T, self.I_fb)
        pot_pr += -self.Ejp * 0.5 * self.alphas[1] * np.exp(2j * np.pi * self.flux) * self.tensor4(self.I_cb, self.e_iphi_op_cb, self.e_iphi_op_cb.T, self.I_fb)
        pot_pr += -self.Ejp * 0.5 * self.alphas[1] * np.exp(-2j * np.pi * self.flux) * self.tensor4(self.I_cb, self.e_iphi_op_cb.T, self.e_iphi_op_cb, self.I_fb)
        pot_pr += -self.Ejp * 0.5 * self.alphas[3] * self.tensor4(self.I_cb, self.e_iphi_op_cb + self.e_iphi_op_cb.T, self.I_cb, self.I_fb)

        return pot_pr

    def kin_p(self):
        self.init_operators()

        C_mat = [
            [(self.alphas[2] + self.alphas[0]) * self.Cjp, 0, -self.alphas[0] * self.Cjp],
            [0, (self.alphas[3] + self.alphas[1]) * self.Cjp, -self.alphas[1] * self.Cjp],
            [-self.alphas[0] * self.Cjp, -self.alphas[1] * self.Cjp, (self.alphas[0] + self.alphas[1]) * self.Cjp + self.Ccp]
        ]

        C_mat_in = np.linalg.inv(C_mat)

        kin_p = C_mat_in[0][0] * self.q1_q1_p
        kin_p += C_mat_in[1][1] * self.q2_q2_p
        kin_p += C_mat_in[2][2] * self.q3_q3_p
        kin_p += 2 * C_mat_in[0][1] * self.q1_q2_p
        kin_p += 2 * C_mat_in[0][2] * self.q1_q3_p
        kin_p += 2 * C_mat_in[1][2] * self.q2_q3_p

        kin_p *= 0.5

        return kin_p

    def pot_p(self):
        self.init_operators()

        pot_p = -self.Ejp * 0.5 * self.alphas[2] * self.tensor3(self.e_iphi_op_cb + self.e_iphi_op_cb.T, self.I_cb, self.I_cb)
        pot_p += -self.Ejp * 0.5 * self.alphas[0] * self.tensor3(self.e_iphi_op_cb.T, self.I_cb, self.e_iphi_op_cb)
        pot_p += -self.Ejp * 0.5 * self.alphas[0] * self.tensor3(self.e_iphi_op_cb, self.I_cb, self.e_iphi_op_cb.T)
        pot_p += -self.Ejp * 0.5 * self.alphas[1] * np.exp(2j * np.pi * self.flux) * self.tensor3(self.I_cb, self.e_iphi_op_cb, self.e_iphi_op_cb.T)
        pot_p += -self.Ejp * 0.5 * self.alphas[1] * np.exp(-2j * np.pi * self.flux) * self.tensor3(self.I_cb, self.e_iphi_op_cb.T, self.e_iphi_op_cb)
        pot_p += -self.Ejp * 0.5 * self.alphas[3] * self.tensor3(self.I_cb, self.e_iphi_op_cb + self.e_iphi_op_cb.T, self.I_cb)
        pot_p += self.Ejp * sum(self.alphas) * self.tensor3(self.I_cb, self.I_cb, self.I_cb)

        return pot_p

    def kin_t(self):
        self.init_operators()

        C_mat = [
            [self.Cct + self.Ct + self.Cjt],
        ]

        C_mat_in = np.linalg.inv(C_mat)

        kin_p = C_mat_in[0][0] * self.q5_q5_t

        kin_p *= 0.5

        return kin_p

    def pot_t(self):
        self.init_operators()

        pot_t = self.Ejt * self.I_cb
        pot_t += -self.Ejt * 0.5 * (self.e_iphi_op_cb + self.e_iphi_op_cb.T)

        return pot_t

    def get_H_pr(self):
        self.H_pr = self.kin_pr() + self.pot_pr()
        # self.H_pr.eliminate_zeros()

        return self.H_pr

    def get_H_prt(self):
        self.H_prt = self.kin_prt() + self.pot_prt()
        # self.H_prt.eliminate_zeros()

        return self.H_prt

    def get_H_p(self):
        self.H_p = self.kin_p() + self.pot_p()
        # self.H_p.eliminate_zeros()

        return self.H_p

    def get_H_t(self):
        self.H_t = self.kin_t() + self.pot_t()
        # self.H_t.eliminate_zeros()

        return self.H_t

    def get_H_p_coupling(self):
        self.init_operators()

        # Raise error if alphas aren't all the same
        if not self.alphas[0] == self.alphas[1] == self.alphas[2] == self.alphas[3]:
            raise Exception('Alphas must be equal to calculate Hq coupling')

        Cr_renorm = (2 * self.alphas[0] * self.Cjp * (self.Ccp + self.Cr) + (1 + self.alphas[0]) * self.Cr * self.Ccp) / (2 * self.alphas[0] * self.Cjp + (1 + self.alphas[0]) * self.Ccp)
        Zr_renorm = np.sqrt(self.Lr / Cr_renorm)
        coeff = (self.hbar / (2 * Zr_renorm))**0.5

        Csum = self.Cr + self.Ccp
        C0 = np.sqrt((1 + self.alphas[0]) * (2 * self.alphas[0] * self.Cjp * Csum + (1 + self.alphas[0]) * self.Cr * self.Ccp))

        self.H_p_coupling = self.alphas[0] * (1 + self.alphas[0]) * (self.Ccp / C0**2) * (self.q1_p + self.q2_p)
        self.H_p_coupling += (1 + self.alphas[0])**2 * (self.Ccp / C0**2) * self.q3_p

        self.H_p_coupling = coeff * self.H_p_coupling

        return self.H_p_coupling

    def diagonalise_prt(self, update=False):
        if update:
            self.H_prt = self.get_H_prt()
        else:
            try:
                self.H_prt
            except AttributeError:
                self.H_prt = self.get_H_prt()

        evals_prt, evecs_prt = sparse.linalg.eigs(
            self.H_prt, k=10, which='SR'
        )
        evecs_prt = evecs_prt.T

        args = np.argsort(evals_prt)
        self.evals_prt = evals_prt[args]
        self.evecs_prt = evecs_prt[args]

        return self.evals_prt, self.evecs_prt

    def diagonalise_p(self, update=False):
        if update:
            self.get_H_p()
        else:
            try:
                self.H_p
            except AttributeError:
                self.get_H_p()

        evals_p, evecs_p = sparse.linalg.eigs(
            self.H_p, 10, which='SR'
        )
        evecs_p = evecs_p.T

        args = np.argsort(evals_p)
        self.evals_p = evals_p[args]
        self.evecs_p = evecs_p[args]

        return self.evals_p, self.evecs_p

    def diagonalise_t(self, update=False):
        if update:
            self.get_H_t()
        else:
            try:
                self.H_t
            except AttributeError:
                self.get_H_t()

        evals_t, evecs_t = sparse.linalg.eigs(
            self.H_t, 5, which='SR'
        )
        evecs_t = evecs_t.T

        args = np.argsort(evals_t)
        self.evals_t = evals_t[args]
        self.evecs_t = evecs_t[args]

        return self.evals_t, self.evecs_t

    def init_probe_states(self, update=False):
        if update:
            self.diagonalise_p(update=True)
        else:
            try:
                self.evecs_p
            except AttributeError:
                self.diagonalise_p()

        self.probe_0 = self.evecs_p[0]
        self.probe_1 = self.evecs_p[1]
        self.probe_minus = 2**-0.5 * (self.probe_0 - self.probe_1)
        self.probe_plus = 2**-0.5 * (self.probe_0 + self.probe_1)

    def init_target_states(self, update=False):
        if update:
            self.diagonalise_t(update=True)
        else:
            try:
                self.evecs_t
            except AttributeError:
                self.diagonalise_t()

        self.target_0 = self.evecs_t[0]
        self.target_1 = self.evecs_t[1]
        self.target_minus = 2**-0.5 * (self.target_0 - self.target_1)
        self.target_plus = 2**-0.5 * (self.target_0 + self.target_1)

    def _plot_probe_states(self):
        self.init_probe_states(update=True)

        plt.figure(figsize=(10, 7))
        plt.title(f'Probe State: |0>, ng: {self.ng}', size=20)
        plt.plot(np.real(self.probe_0))
        plt.plot(np.imag(self.probe_0))
        plt.show()

        plt.figure(figsize=(10, 7))
        plt.title(f'Probe State: |1>, ng: {self.ng}', size=20)
        plt.plot(np.real(self.probe_1))
        plt.plot(np.imag(self.probe_1))
        plt.show()

        plt.figure(figsize=(10, 7))
        plt.title(f'Probe State: |+>, ng: {self.ng}', size=20)
        plt.plot(np.real(self.probe_plus))
        plt.plot(np.imag(self.probe_plus))
        plt.show()

        plt.figure(figsize=(10, 7))
        plt.title(f'Probe State: |->, ng: {self.ng}', size=20)
        plt.plot(np.real(self.probe_minus))
        plt.plot(np.imag(self.probe_minus))
        plt.show()

    def init_cavity_states(self):
        self.fock_0 = np.zeros((self.mcut,))
        self.fock_0[0] = 1

        self.fock_1 = np.zeros((self.mcut,))
        self.fock_1[1] = 1

        self.fock_m = 2**-0.5 * (self.fock_0 - self.fock_1)
        self.fock_p = 2**-0.5 * (self.fock_0 + self.fock_1)

    def init_prod_states(self, update=False):
        try:
            self.fock_0
        except AttributeError:
            self.init_cavity_states()

        if update:
            self.init_probe_states(update=True)
        else:
            try:
                self.probe_0
            except AttributeError:
                self.init_probe_states()

        if update:
            self.init_target_states(update=True)
        else:
            try:
                self.target_0
            except AttributeError:
                self.init_target_states()

        self.ket_00_ = np.kron(self.probe_0, self.fock_0)
        self.ket_10_ = np.kron(self.probe_1, self.fock_0)

        self.ket_000 = np.kron(np.kron(self.probe_0, self.fock_0), self.target_0)
        self.ket_001 = np.kron(np.kron(self.probe_0, self.fock_0), self.target_1)
        self.ket_010 = np.kron(np.kron(self.probe_0, self.fock_1), self.target_0)
        self.ket_100 = np.kron(np.kron(self.probe_1, self.fock_0), self.target_0)
        self.ket_011 = np.kron(np.kron(self.probe_0, self.fock_1), self.target_1)
        self.ket_110 = np.kron(np.kron(self.probe_1, self.fock_1), self.target_0)
        self.ket_p10 = np.kron(np.kron(self.probe_plus, self.fock_1), self.target_0)
        self.ket_p00 = np.kron(np.kron(self.probe_plus, self.fock_0), self.target_0)
        self.ket_m00 = np.kron(np.kron(self.probe_minus, self.fock_0), self.target_0)

    def calc_g_parr(self, update=False):
        if update:
            self.H_prt = self.get_H_prt()
            self.init_prod_states(update=True)
        else:
            try:
                self.H_prt
            except AttributeError:
                self.H_prt = self.get_H_prt()

            try:
                self.ket_p10
                self.ket_m00
            except AttributeError:
                self.init_prod_states()

        g_parr = 1j * self.hc(self.ket_p10) @ self.H_prt @ self.ket_m00

        return g_parr

    def calc_g_perp(self, update=False):
        if update:
            self.H_prt = self.get_H_prt()
            self.init_prod_states(update=True)
        else:
            try:
                self.H_prt
            except AttributeError:
                self.H_prt = self.get_H_prt()

            try:
                self.ket_110
                self.ket_000
            except AttributeError:
                self.init_prod_states()

        g_perp = self.hc(self.ket_110) @ self.H_prt @ self.ket_000

        return g_perp

    def calc_probe_freq(self, update=False):
        if update:
            self.H_prt = self.get_H_prt()
            self.init_prod_states(update=True)
        else:
            try:
                self.H_prt
            except AttributeError:
                self.H_prt = self.get_H_prt()

            try:
                self.ket_p1
            except AttributeError:
                self.init_prod_states()

        return np.abs(self.hc(self.ket_100) @ self.H_prt @ self.ket_100) - np.abs(self.hc(self.ket_000) @ self.H_prt @ self.ket_000)

    def calc_cavity_freq(self, update=False):
        if update:
            self.H_prt = self.get_H_prt()
            self.init_prod_states(update=True)
        else:
            try:
                self.H_prt
            except AttributeError:
                self.H_prt = self.get_H_prt()

            try:
                self.ket_p1
            except AttributeError:
                self.init_prod_states()

        return self.hc(self.ket_010) @ self.H_prt @ self.ket_010 - self.hc(self.ket_000) @ self.H_prt @ self.ket_000

    def calc_target_freq(self, update=False):
        if update:
            self.H_prt = self.get_H_prt()
            self.init_prod_states(update=True)
        else:
            try:
                self.H_prt
            except AttributeError:
                self.H_prt = self.get_H_prt()

            try:
                self.ket_p1
            except AttributeError:
                self.init_prod_states()

        return self.hc(self.ket_001) @ self.H_prt @ self.ket_001 - self.hc(self.ket_000) @ self.H_prt @ self.ket_000

    def calc_bil_omega(self):
        Cr_renorm = (2 * self.alphas[0] * self.Cjp * (self.Ccp + self.Cr) + (1 + self.alphas[0]) * self.Cr * self.Ccp) / (2 * self.alphas[0] * self.Cjp + (1 + self.alphas[0]) * self.Ccp)
        omega_exp = self.hbar / np.sqrt(self.Lr * Cr_renorm)

        return omega_exp

    def calc_bil_coupling(self, update=False):
        C0 = np.sqrt((1 + self.alphas[0]) * (2 * self.alphas[0] * self.Cjp * (self.Cr + self.Ccp) + (1 + self.alphas[0]) * self.Cr * self.Ccp))

        Cr_renorm = (2 * self.alphas[0] * self.Cjp * (self.Ccp + self.Cr) + (1 + self.alphas[0]) * self.Cr * self.Ccp) / (2 * self.alphas[0] * self.Cjp + (1 + self.alphas[0]) * self.Ccp)
        Zr_renorm = np.sqrt(self.Lr / Cr_renorm)
        coeff = np.sqrt(self.hbar / (2 * Zr_renorm))

        coupling_op = self.alphas[0] * (1 + self.alphas[0]) * (self.Ccp / C0**2) * (self.q1_p + self.q2_p) + (1 + self.alphas[0])**2 * (self.Ccp / C0**2) * self.q3_p

        if update:
            self.init_probe_states(update=True)
        else:
            try:
                self.probe_0
            except AttributeError:
                self.init_probe_states()

        g_parr_bil = -1 * coeff * (self.hc(self.probe_plus) @ coupling_op @ self.probe_minus)
        g_perp_bil = coeff * (self.hc(self.probe_1) @ coupling_op @ self.probe_0)

        return g_parr_bil, g_perp_bil
