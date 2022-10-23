import numpy as np
import scipy as sp
import scipy.sparse.linalg
import scipy.sparse as sparse


class RCCircuit:
    
    def __init__(self, EJ=121e9, CJ=8e-15, Cr=100e-15, Cc=5e-15, Lr=10e-9, alphas=[0.4]*4, ng=0.25, flux=0.5, ncut=2, mcut=20):
        self.h = 6.626e-34
        self.hbar = 1.055e-34
        self.e_charge = 1.60218e-19
        
        self.EJ = EJ * self.h                   # Josephson energy
        self.CJ = CJ                            # Josephson capacitance
        self.Cr = Cr                            # Resonator capacitance
        self.Cc = Cc                            # Coupling capacitance
        self.Lr = Lr                            # Resonator inductance
        self.alphas = alphas                    # Asymmetry of flux qubit
        self.ng = ng                            # Reduced gate charge
        self.flux = flux                        # Flux through qubit
        self.Z0 = np.sqrt(self.Lr / self.Cr)    # Impedance of resonator
        self.ncut = ncut                        # Cut-off threshold for number basis
        self.mcut = mcut                        # Cut-off threshold for resonator Fock states
        
        self.init_operators()
        
    def _print_params(self):
        print(f'EJ:    {self.EJ}')
        print(f'CJ:    {self.CJ}')
        print(f'Cr:    {self.Cr}')
        print(f'Cc:    {self.Cc}')
        print(f'Lr:    {self.Lr}')
        print(f'alpha: {self.alpha}')
        print(f'ng:    {self.ng}')
        print(f'flux:  {self.flux}')

    def init_operators(self):
        self.I_cb = sp.sparse.diags(np.ones(2 * self.ncut + 1))   # Identity for qubit (charge basis)
        self.I_fb = sp.sparse.diags(np.ones(self.mcut))           # Identity for resonator (Fock basis)
        
        self.q_op_cb = sp.sparse.diags(-2 * self.e_charge * np.arange(-self.ncut, self.ncut + 1, dtype=np.complex_))           # Charge operator (charge basis)
        self.ng_op_cb = -2 * self.e_charge * self.ng * self.I_cb
        self.e_iphi_op_cb = sp.sparse.diags(np.ones(2 * self.ncut, dtype=np.complex_), offsets=1)                                   # e^{i \phi} operator (charge basis)
        self.creation_op_fb = sp.sparse.diags(np.sqrt(np.arange(1, self.mcut, dtype=np.complex_)), offsets=-1)                      # Creation operator (Fock basis)
        self.annihilation_op_fb = np.conjugate(self.creation_op_fb.T)                                                 # Annihilation operator (Fock basis)
        self.q_op_fb = np.sqrt(self.hbar / (2 * self.Z0)) * 1j * (self.creation_op_fb - self.annihilation_op_fb)      # Charge operator (Fock basis)
        self.phi_op_fb = np.sqrt(self.hbar * self.Z0 * 0.5) * (self.creation_op_fb + self.annihilation_op_fb)         # Phi operator (Fock basis)
        
        self.q1_q1_qr = self.tensor4(self.q_op_cb @ self.q_op_cb, self.I_cb, self.I_cb, self.I_fb)
        self.q1_q2_qr = self.tensor4(self.q_op_cb, self.q_op_cb, self.I_cb, self.I_fb)
        self.q1_q3_qr = self.tensor4(self.q_op_cb, self.I_cb, self.q_op_cb + self.ng_op_cb, self.I_fb)
        self.q1_q4_qr = self.tensor4(self.q_op_cb, self.I_cb, self.I_cb, self.q_op_fb)
        self.q2_q2_qr = self.tensor4(self.I_cb, self.q_op_cb @ self.q_op_cb, self.I_cb, self.I_fb)
        self.q2_q3_qr = self.tensor4(self.I_cb, self.q_op_cb, self.q_op_cb + self.ng_op_cb, self.I_fb)
        self.q2_q4_qr = self.tensor4(self.I_cb, self.q_op_cb, self.I_cb, self.q_op_fb)
        self.q3_q3_qr = self.tensor4(self.I_cb, self.I_cb, (self.q_op_cb + self.ng_op_cb) @ (self.q_op_cb + self.ng_op_cb), self.I_fb)
        self.q3_q4_qr = self.tensor4(self.I_cb, self.I_cb, self.q_op_cb + self.ng_op_cb, self.q_op_fb)
        self.q4_q4_qr = self.tensor4(self.I_cb, self.I_cb, self.I_cb, self.q_op_fb @ self.q_op_fb)
        
        self.q1_q = self.tensor3(self.q_op_cb, self.I_cb, self.I_cb)
        self.q2_q = self.tensor3(self.I_cb, self.q_op_cb, self.I_cb)
        self.q3_q = self.tensor3(self.I_cb, self.I_cb, self.q_op_cb + self.ng_op_cb)
        
        self.q1_q1_q = self.tensor3(self.q_op_cb @ self.q_op_cb, self.I_cb, self.I_cb)
        self.q1_q2_q = self.tensor3(self.q_op_cb, self.q_op_cb, self.I_cb)
        self.q1_q3_q = self.tensor3(self.q_op_cb, self.I_cb, self.q_op_cb + self.ng_op_cb)
        self.q2_q2_q = self.tensor3(self.I_cb, self.q_op_cb @ self.q_op_cb, self.I_cb)
        self.q2_q3_q = self.tensor3(self.I_cb, self.q_op_cb, self.q_op_cb + self.ng_op_cb)
        self.q3_q3_q = self.tensor3(self.I_cb, self.I_cb, (self.q_op_cb + self.ng_op_cb) @ (self.q_op_cb + self.ng_op_cb))
    
    def tensor3(self, op1, op2, op3):
        return sparse.kron(sparse.kron(op1, op2), op3)
    
    def tensor4(self, op1, op2, op3, op4):
        return sparse.kron(sparse.kron(sparse.kron(op1, op2), op3), op4)
    
    def hc(self, state):
        return np.conjugate(state).T
        
    def kin_qr(self):
        self.init_operators()
        
        C_mat = [
            [(self.alphas[2] + self.alphas[0]) * self.CJ, 0, -self.alphas[0] * self.CJ, 0],
            [0, (self.alphas[3] + self.alphas[1]) * self.CJ, -self.alphas[1] * self.CJ, 0],
            [-self.alphas[0] * self.CJ, -self.alphas[1] * self.CJ, (self.alphas[0] + self.alphas[1]) * self.CJ + self.Cc, -self.Cc],
            [0, 0, -self.Cc, self.Cr + self.Cc]
        ]
        
        C_mat_in = np.linalg.inv(C_mat)

        kin_qr = C_mat_in[0][0] * self.q1_q1_qr
        kin_qr += C_mat_in[1][1] * self.q2_q2_qr
        kin_qr += C_mat_in[2][2] * self.q3_q3_qr
        kin_qr += C_mat_in[3][3] * self.q4_q4_qr
        kin_qr += 2 * C_mat_in[0][1] * self.q1_q2_qr
        kin_qr += 2 * C_mat_in[0][2] * self.q1_q3_qr
        kin_qr += 2 * C_mat_in[0][3] * self.q1_q4_qr
        kin_qr += 2 * C_mat_in[1][2] * self.q2_q3_qr
        kin_qr += 2 * C_mat_in[1][3] * self.q2_q4_qr
        kin_qr += 2 * C_mat_in[2][3] * self.q3_q4_qr
        
        kin_qr *= 0.5

        return kin_qr

    def pot_qr(self):
        self.init_operators()
        
        pot_qr = -self.EJ * 0.5 * self.alphas[2] * self.tensor4(self.e_iphi_op_cb + self.e_iphi_op_cb.T, self.I_cb, self.I_cb, self.I_fb)
        pot_qr += -self.EJ * 0.5 * self.alphas[0] * self.tensor4(self.e_iphi_op_cb.T, self.I_cb, self.e_iphi_op_cb, self.I_fb)
        pot_qr += -self.EJ * 0.5 * self.alphas[0] * self.tensor4(self.e_iphi_op_cb, self.I_cb, self.e_iphi_op_cb.T, self.I_fb)
        pot_qr += -self.EJ * 0.5 * self.alphas[1] * np.exp(2j * np.pi * self.flux) * self.tensor4(self.I_cb, self.e_iphi_op_cb, self.e_iphi_op_cb.T, self.I_fb)
        pot_qr += -self.EJ * 0.5 * self.alphas[1] * np.exp(-2j * np.pi * self.flux) * self.tensor4(self.I_cb, self.e_iphi_op_cb.T, self.e_iphi_op_cb, self.I_fb)
        pot_qr += -self.EJ * 0.5 * self.alphas[3] * self.tensor4(self.I_cb, self.e_iphi_op_cb + self.e_iphi_op_cb.T, self.I_cb, self.I_fb)
        pot_qr += self.EJ * sum(self.alphas) * self.tensor4(self.I_cb, self.I_cb, self.I_cb, self.I_fb)
        pot_qr += 0.5 * self.Lr**-1 * self.tensor4(self.I_cb, self.I_cb, self.I_cb, self.phi_op_fb @ self.phi_op_fb)
        
        return pot_qr
        
    def kin_q(self):
        self.init_operators()
        
        C_mat = [
            [(self.alphas[2] + self.alphas[0]) * self.CJ, 0, -self.alphas[0] * self.CJ],
            [0, (self.alphas[3] + self.alphas[1]) * self.CJ, -self.alphas[1] * self.CJ],
            [-self.alphas[0] * self.CJ, -self.alphas[1] * self.CJ, (self.alphas[0] + self.alphas[1]) * self.CJ + self.Cc]
        ]
        
        C_mat_in = np.linalg.inv(C_mat)

        kin_q = C_mat_in[0][0] * self.q1_q1_q
        kin_q += C_mat_in[1][1] * self.q2_q2_q
        kin_q += C_mat_in[2][2] * self.q3_q3_q
        kin_q += 2 * C_mat_in[0][1] * self.q1_q2_q
        kin_q += 2 * C_mat_in[0][2] * self.q1_q3_q
        kin_q += 2 * C_mat_in[1][2] * self.q2_q3_q
        
        kin_q *= 0.5
        
        return kin_q

    def pot_q(self):
        self.init_operators()

        pot_q = -self.EJ * 0.5 * self.alphas[2] * self.tensor3(self.e_iphi_op_cb + self.e_iphi_op_cb.T, self.I_cb, self.I_cb)
        pot_q += -self.EJ * 0.5 * self.alphas[0] * self.tensor3(self.e_iphi_op_cb.T, self.I_cb, self.e_iphi_op_cb)
        pot_q += -self.EJ * 0.5 * self.alphas[0] * self.tensor3(self.e_iphi_op_cb, self.I_cb, self.e_iphi_op_cb.T)
        pot_q += -self.EJ * 0.5 * self.alphas[1] * np.exp(2j * np.pi * self.flux) * self.tensor3(self.I_cb, self.e_iphi_op_cb, self.e_iphi_op_cb.T)
        pot_q += -self.EJ * 0.5 * self.alphas[1] * np.exp(-2j * np.pi * self.flux) * self.tensor3(self.I_cb, self.e_iphi_op_cb.T, self.e_iphi_op_cb)
        pot_q += -self.EJ * 0.5 * self.alphas[3] * self.tensor3(self.I_cb, self.e_iphi_op_cb + self.e_iphi_op_cb.T, self.I_cb)
        pot_q += self.EJ * sum(self.alphas) * self.tensor3(self.I_cb, self.I_cb, self.I_cb)
        
        return pot_q

    def get_H_qr(self):
        self.H_qr = self.kin_qr() + self.pot_qr()
        self.H_qr.eliminate_zeros()
        
        return self.H_qr

    def get_H_q(self):
        self.H_q = self.kin_q() + self.pot_q()
        self.H_q.eliminate_zeros()
        
        return self.H_q

    def get_H_q_coupling(self):
        self.init_operators()
        
        Cr_renorm = (2 * self.alpha * self.CJ * (self.Cc + self.Cr) + (1 + self.alpha) * self.Cr * self.Cc) / (2 * self.alpha * self.CJ + (1 + self.alpha) * self.Cc)
        Zr_renorm = np.sqrt(self.Lr / Cr_renorm)
        coeff = (self.hbar / (2 * Zr_renorm))**0.5

        Csum = self.Cr + self.Cc
        Cstar = (self.Cr * self.Cc) / self.CJ
        C0 = np.sqrt((1 + self.alpha) * (2 * self.alpha * self.CJ * Csum + (1 + self.alpha) * self.Cr * self.Cc))
        
        self.H_q_coupling = self.alpha * (1 + self.alpha) * (self.Cc / C0**2) * (self.q1_q + self.q2_q)
        self.H_q_coupling += (1 + self.alpha)**2 * (self.Cc / C0**2) * self.q3_q
        
        self.H_q_coupling = coeff * self.H_q_coupling
        
        return self.H_q_coupling
    
    def diagonalise_qr(self, update=False):
        if update:
            self.get_H_qr()
        else:
            try:
                self.H_qr
            except AttributeError:
                self.get_H_qr()
        
        evals_qr, evecs_qr = sparse.linalg.eigs(
            self.H_qr, k=10, which='SR'
        )
        evecs_qr = evecs_qr.T
        
        args = np.argsort(evals_qr)
        self.evals_qr = evals_qr[args]
        self.evecs_qr = evecs_qr[args]
        
        return self.evals_qr, self.evecs_qr
    
    def diagonalise_q(self, update=False):
        if update:
            self.get_H_q()
        else:
            try:
                self.H_q
            except AttributeError:
                self.get_H_q()
        
        evals_q, evecs_q = sparse.linalg.eigs(
            self.H_q, 10, which='SR'
        )
        evecs_q = evecs_q.T
        
        args = np.argsort(evals_q)
        self.evals_q = evals_q[args]
        self.evecs_q = evecs_q[args]
        
        return self.evals_q, self.evecs_q
    
    def init_qubit_states(self, update=False):
        if update:
            self.diagonalise_q(update=True)
        else:
            try:
                self.evecs_q
            except AttributeError:
                self.diagonalise_q()

        self.qubit_0 = self.evecs_q[0]
        self.qubit_1 = self.evecs_q[1]
        self.qubit_m = 2**-0.5 * (self.qubit_0 - self.qubit_1)
        self.qubit_p = 2**-0.5 * (self.qubit_0 + self.qubit_1)
    
    def _plot_qubit_states(self):
        self.init_qubit_states(update=True)
        
        plt.figure(figsize=(10, 7))
        plt.title(f'Qubit State: |0>, ng: {self.ng}', size=20)
        plt.plot(np.real(self.qubit_0))
        plt.plot(np.imag(self.qubit_0))
        plt.show()
        
        plt.figure(figsize=(10, 7))
        plt.title(f'Qubit State: |1>, ng: {self.ng}', size=20)
        plt.plot(np.real(self.qubit_1))
        plt.plot(np.imag(self.qubit_1))
        plt.show()
        
        plt.figure(figsize=(10, 7))
        plt.title(f'Qubit State: |+>, ng: {self.ng}', size=20)
        plt.plot(np.real(self.qubit_p))
        plt.plot(np.imag(self.qubit_p))
        plt.show()
        
        plt.figure(figsize=(10, 7))
        plt.title(f'Qubit State: |->, ng: {self.ng}', size=20)
        plt.plot(np.real(self.qubit_m))
        plt.plot(np.imag(self.qubit_m))
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
            self.init_qubit_states(update=True)
        else:
            try:
                self.qubit_0
            except AttributeError:
                self.init_qubit_states()

        self.ket_00 = np.kron(self.qubit_0, self.fock_0)
        self.ket_01 = np.kron(self.qubit_0, self.fock_1)
        self.ket_10 = np.kron(self.qubit_1, self.fock_0)
        self.ket_11 = np.kron(self.qubit_1, self.fock_1)
        self.ket_0m = np.kron(self.qubit_0, self.fock_m)
        self.ket_0p = np.kron(self.qubit_0, self.fock_p)
        self.ket_p1 = np.kron(self.qubit_p, self.fock_1)
        self.ket_m0 = np.kron(self.qubit_m, self.fock_0)
    
    def calc_g_parr(self, update=False):
        if update:
            self.get_H_qr()
            self.init_prod_states(update=True)
        else:
            try:
                self.H_qr
            except AttributeError:
                self.get_H_qr()

            try:
                self.ket_p1
            except AttributeError:
                self.init_prod_states()
        
        g_parr = 1j * self.hc(self.ket_p1) @ self.H_qr @ self.ket_m0
        
        return g_parr
    
    def calc_g_perp(self, update=False):
        if update:
            self.get_H_qr()
            self.init_prod_states(update=True)
        else:
            try:
                self.H_qr
            except AttributeError:
                self.get_H_qr()

            try:
                self.ket_p1
            except AttributeError:
                self.init_prod_states()
        
        g_perp = self.hc(self.ket_11) @ self.H_qr @ self.ket_00
        
        return g_perp
    
    def calc_delta(self, update=False):
        if update:
            self.get_H_qr()
            self.init_prod_states(update=True)
        else:
            try:
                self.H_qr
            except AttributeError:
                self.get_H_qr()

            try:
                self.ket_p1
            except AttributeError:
                self.init_prod_states()
        
        delta = self.hc(self.ket_10) @ self.H_qr @ self.ket_10 - self.hc(self.ket_00) @ self.H_qr @ self.ket_00
        
        return delta
    
    def calc_omega(self, update=False):
        if update:
            self.get_H_qr()
            self.init_prod_states(update=True)
        else:
            try:
                self.H_qr
            except AttributeError:
                self.get_H_qr()

            try:
                self.ket_p1
            except AttributeError:
                self.init_prod_states()
        
        omega = self.hc(self.ket_01) @ self.H_qr @ self.ket_01 - 0.5 * self.calc_delta()
        
        return omega
    
    def calc_bil_omega(self):
        Cr_renorm = (2 * self.alpha * self.CJ * (self.Cc + self.Cr) + (1 + self.alpha) * self.Cr * self.Cc) / (2 * self.alpha * self.CJ + (1 + self.alpha) * self.Cc)
        omega_exp = self.hbar / np.sqrt(self.Lr * Cr_renorm)
        
        return omega_exp
    
    def calc_bil_coupling(self, update=False):
        C0 = np.sqrt((1 + self.alpha) * (2 * self.alpha * self.CJ * (self.Cr + self.Cc) + (1 + self.alpha) * self.Cr * self.Cc))

        Cr_renorm = (2 * self.alpha * self.CJ * (self.Cc + self.Cr) + (1 + self.alpha) * self.Cr * self.Cc) / (2 * self.alpha * self.CJ + (1 + self.alpha) * self.Cc)
        Zr_renorm = np.sqrt(self.Lr / Cr_renorm)
        coeff = np.sqrt(self.hbar / (2 * Zr_renorm))

        coupling_op = self.alpha * (1 + self.alpha) * (self.Cc / C0**2) * (self.q1_q + self.q2_q) + (1 + self.alpha)**2 * (self.Cc / C0**2) * self.q3_q

        if update:
            self.init_qubit_states(update=True)
        else:
            try:
                self.qubit_0
            except AttributeError:
                self.init_qubit_states()

        g_parr_bil = -1 * coeff * (self.hc(self.qubit_p) @ coupling_op @ self.qubit_m)
        g_perp_bil = coeff * (self.hc(self.qubit_1) @ coupling_op @ self.qubit_0)
        
        return g_parr_bil, g_perp_bil

class RCCircuitAlphaDisp:
    def __init__(self, Ejp=153e9, Ejt=21.3e9, Cjp=12e-15, Cjt=36e-15, Cc=3.1e-15, Ct=18e-15, alphas=[0.25]*2, ng=0.12, flux=0.5, ncut=2):
        self.h = 6.626e-34
        self.hbar = 1.055e-34
        self.e_charge = 1.60218e-19
        
        self.Ejp = Ejp * self.h                 # Josephson energy of probe
        self.Ejt = Ejt * self.h                 # Josephson energy of target
        self.Cjp = Cjp                          # Josephson capacitance of probe
        self.Cjt = Cjt                          # Josephson capacitance of target
        self.Ct = Ct                            # Target Capacitor
        self.Cc = Cc                            # Coupling capacitance
        self.ng = ng                            # Reduced gate charge
        self.flux = flux                        # Flux through qubit
        self.ncut = ncut                        # Cut-off threshold for number basis
        self.alphas = alphas
        
        self.init_operators()
        
    def _print_params(self):
        print(f'EJ:    {self.EJ}')
        print(f'CJ:    {self.CJ}')
        print(f'Cr:    {self.Cr}')
        print(f'Cc:    {self.Cc}')
        print(f'Lr:    {self.Lr}')
        print(f'alphas: {self.alphas}')
        print(f'ng:    {self.ng}')
        print(f'flux:  {self.flux}')

    def init_operators(self):
        self.I_cb = sp.sparse.diags(np.ones(2 * self.ncut + 1))   # Identity for qubit (charge basis)
        
        self.q_op_cb = sp.sparse.diags(2 * self.e_charge * np.arange(-self.ncut, self.ncut + 1, dtype=np.complex_))           # Charge operator (charge basis)
        self.ng_op_cb = 2 * self.e_charge * self.ng * self.I_cb
        self.e_iphi_op_cb = sp.sparse.diags(np.ones(2 * self.ncut, dtype=np.complex_), offsets=1)
        
        self.q1_q1_pt = self.tensor4(self.q_op_cb @ self.q_op_cb, self.I_cb, self.I_cb, self.I_cb)
        self.q1_q2_pt = self.tensor4(self.q_op_cb, self.q_op_cb, self.I_cb, self.I_cb)
        self.q1_q3_pt = self.tensor4(self.q_op_cb, self.I_cb, self.q_op_cb + self.ng_op_cb, self.I_cb)
        self.q1_q4_pt = self.tensor4(self.q_op_cb, self.I_cb, self.I_cb, self.q_op_cb)
        self.q2_q2_pt = self.tensor4(self.I_cb, self.q_op_cb @ self.q_op_cb, self.I_cb, self.I_cb)
        self.q2_q3_pt = self.tensor4(self.I_cb, self.q_op_cb, self.q_op_cb + self.ng_op_cb, self.I_cb)
        self.q2_q4_pt = self.tensor4(self.I_cb, self.q_op_cb, self.I_cb, self.q_op_cb)
        self.q3_q3_pt = self.tensor4(self.I_cb, self.I_cb, (self.q_op_cb + self.ng_op_cb) @ (self.q_op_cb + self.ng_op_cb), self.I_cb)
        self.q3_q4_pt = self.tensor4(self.I_cb, self.I_cb, self.q_op_cb + self.ng_op_cb, self.q_op_cb)
        self.q4_q4_pt = self.tensor4(self.I_cb, self.I_cb, self.I_cb, self.q_op_cb @ self.q_op_cb)
        
        self.q1_p = self.tensor3(self.q_op_cb, self.I_cb, self.I_cb)
        self.q2_p = self.tensor3(self.I_cb, self.q_op_cb, self.I_cb)
        self.q3_p = self.tensor3(self.I_cb, self.I_cb, self.q_op_cb + self.ng_op_cb)
        
        self.q1_q1_p = self.tensor3(self.q_op_cb @ self.q_op_cb, self.I_cb, self.I_cb)
        self.q1_q2_p = self.tensor3(self.q_op_cb, self.q_op_cb, self.I_cb)
        self.q1_q3_p = self.tensor3(self.q_op_cb, self.I_cb, self.q_op_cb + self.ng_op_cb)
        self.q2_q2_p = self.tensor3(self.I_cb, self.q_op_cb @ self.q_op_cb, self.I_cb)
        self.q2_q3_p = self.tensor3(self.I_cb, self.q_op_cb, self.q_op_cb + self.ng_op_cb)
        self.q3_q3_p = self.tensor3(self.I_cb, self.I_cb, (self.q_op_cb + self.ng_op_cb) @ (self.q_op_cb + self.ng_op_cb))
    
    def tensor3(self, op1, op2, op3):
        return sparse.kron(sparse.kron(op1, op2), op3)
    
    def tensor4(self, op1, op2, op3, op4):
        return sparse.kron(sparse.kron(sparse.kron(op1, op2), op3), op4)
    
    def hc(self, state):
        return np.conjugate(state).T
        
    def get_kin_pt(self):
        self.init_operators()
        
        C_mat = [
            [(1 + self.alphas[0]) * self.Cjp, 0, -self.alphas[0] * self.Cjp, 0],
            [0, (1 + self.alphas[1]) * self.Cjp, -self.alphas[1] * self.Cjp, 0],
            [-self.alphas[0] * self.Cjp, -self.alphas[1] * self.Cjp, sum(self.alphas) * self.Cjp + self.Cc, -self.Cc],
            [0, 0, -self.Cc, self.Cjt + self.Cc + self.Ct]
        ]
        
        C_mat_in = np.linalg.inv(C_mat)

        kin_pt = C_mat_in[0][0] * self.q1_q1_pt
        kin_pt += C_mat_in[1][1] * self.q2_q2_pt
        kin_pt += C_mat_in[2][2] * self.q3_q3_pt
        kin_pt += C_mat_in[3][3] * self.q4_q4_pt
        kin_pt += 2 * C_mat_in[0][1] * self.q1_q2_pt
        kin_pt += 2 * C_mat_in[0][2] * self.q1_q3_pt
        kin_pt += 2 * C_mat_in[0][3] * self.q1_q4_pt
        kin_pt += 2 * C_mat_in[1][2] * self.q2_q3_pt
        kin_pt += 2 * C_mat_in[1][3] * self.q2_q4_pt
        kin_pt += 2 * C_mat_in[2][3] * self.q3_q4_pt
        
        kin_pt *= 0.5

        return kin_pt

    def get_pot_pt(self):
        self.init_operators()
        
        pot_pt = -self.Ejp * 0.5 * self.tensor4(self.e_iphi_op_cb + self.e_iphi_op_cb.T, self.I_cb, self.I_cb, self.I_cb)
        pot_pt += -self.Ejp * 0.5 * self.tensor4(self.I_cb, self.e_iphi_op_cb + self.e_iphi_op_cb.T, self.I_cb, self.I_cb)
        pot_pt += -self.Ejp * 0.5 * self.alphas[0] * self.tensor4(self.e_iphi_op_cb.T, self.I_cb, self.e_iphi_op_cb, self.I_cb)
        pot_pt += -self.Ejp * 0.5 * self.alphas[0] * self.tensor4(self.e_iphi_op_cb, self.I_cb, self.e_iphi_op_cb.T, self.I_cb)
        pot_pt += -self.Ejp * 0.5 * self.alphas[1] * np.exp(2j * np.pi * self.flux) * self.tensor4(self.I_cb, self.e_iphi_op_cb, self.e_iphi_op_cb.T, self.I_cb)
        pot_pt += -self.Ejp * 0.5 * self.alphas[1] * np.exp(-2j * np.pi * self.flux) * self.tensor4(self.I_cb, self.e_iphi_op_cb.T, self.e_iphi_op_cb, self.I_cb)
        pot_pt += self.Ejp * (sum(self.alphas) + 2) * self.tensor4(self.I_cb, self.I_cb, self.I_cb, self.I_cb)
        
        pot_pt += self.Ejt * self.tensor4(self.I_cb, self.I_cb, self.I_cb, self.I_cb)        
        pot_pt += -self.Ejt * self.tensor4(self.I_cb, self.I_cb, self.I_cb, self.e_iphi_op_cb + self.e_iphi_op_cb.T)
        
        return pot_pt
        
    def get_kin_p(self):
        self.init_operators()
        
        C_mat = [
            [(1 + self.alphas[0]) * self.Cjp, 0, -self.alphas[0] * self.Cjp],
            [0, (1 + self.alphas[1]) * self.Cjp, -self.alphas[1] * self.Cjp],
            [-self.alphas[0] * self.Cjp, -self.alphas[1] * self.Cjp, sum(self.alphas) * self.Cjp + self.Cc]
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

    def get_pot_p(self):
        self.init_operators()

        pot_p = -self.Ejp * 0.5 * self.tensor3(self.e_iphi_op_cb + self.e_iphi_op_cb.T, self.I_cb, self.I_cb)
        pot_p += -self.Ejp * 0.5 * self.alphas[0] * self.tensor3(self.e_iphi_op_cb.T, self.I_cb, self.e_iphi_op_cb)
        pot_p += -self.Ejp * 0.5 * self.alphas[0] * self.tensor3(self.e_iphi_op_cb, self.I_cb, self.e_iphi_op_cb.T)
        pot_p += -self.Ejp * 0.5 * self.alphas[1] * np.exp(2j * np.pi * self.flux) * self.tensor3(self.I_cb, self.e_iphi_op_cb, self.e_iphi_op_cb.T)
        pot_p += -self.Ejp * 0.5 * self.alphas[1] * np.exp(-2j * np.pi * self.flux) * self.tensor3(self.I_cb, self.e_iphi_op_cb.T, self.e_iphi_op_cb)
        pot_p += -self.Ejp * 0.5 * self.tensor3(self.I_cb, self.e_iphi_op_cb + self.e_iphi_op_cb.T, self.I_cb)
        pot_p += self.Ejp * (2 + sum(self.alphas)) * self.tensor3(self.I_cb, self.I_cb, self.I_cb)
        
        return pot_p
        
    def get_kin_t(self):
        self.init_operators()
        
        kin_t = 0.5 * (self.q_op_cb @ self.q_op_cb) / (self.Cjt + self.Ct + self.Cc)

        return kin_t

    def get_pot_t(self):
        self.init_operators()
        
        pot_t = self.Ejt * self.I_cb
        pot_t += -self.Ejt * (self.e_iphi_op_cb + self.e_iphi_op_cb.T)
        
        return pot_t

    def get_H_pt(self):
        self.H_pt = self.get_kin_pt() + self.get_pot_pt()
        self.H_pt.eliminate_zeros()
        
        return self.H_pt

    def get_H_p(self):
        self.H_p = self.get_kin_p() + self.get_pot_p()
        self.H_p.eliminate_zeros()
        
        return self.H_p

    def get_H_t(self):
        self.H_t = self.get_kin_t() + self.get_pot_t()
        
        return self.H_t
    
    def diagonalise_pt(self, update=False):
        if update:
            self.get_H_pt()
        else:
            try:
                self.H_pt
            except AttributeError:
                self.get_H_pt()
        
        self.evals_pt, evecs = sparse.linalg.eigs(
            self.H_pt, k=10, which='SR'
        )
        
        evecs = evecs.T
        self.evecs_pt = [
            sp.sparse.csr_array(evecs[0]),
            sp.sparse.csr_array(evecs[1]),
        ]
        
        return self.evals_pt, self.evecs_pt
    
    def diagonalise_p(self, update=False):
        if update:
            self.get_H_p()
        else:
            try:
                self.H_p
            except AttributeError:
                self.get_H_p()
        
        self.evals_p, evecs = sparse.linalg.eigs(
            self.H_p, k=10, which='SR'
        )
        
        evecs = evecs.T
        self.evecs_p = [
            sp.sparse.csr_array(evecs[0]),
            sp.sparse.csr_array(evecs[1]),
        ]
        
        return self.evals_p, self.evecs_p
    
    def diagonalise_t(self, update=False):
        if update:
            self.get_H_t()
        else:
            try:
                self.H_t
            except AttributeError:
                self.get_H_t()
        
        self.evals_t, evecs = sparse.linalg.eigs(
            self.H_t, k=2, which='SR'
        )
        
        evecs = evecs.T
        self.evecs_t = [
            sp.sparse.csr_array(evecs[0]),
            sp.sparse.csr_array(evecs[1]),
        ]
        
        return self.evals_t, self.evecs_t
    
    def init_probe_states(self, update=False):
        if update:
            self.diagonalise_p(update=True)
            self.diagonalise_t(update=True)
        else:
            try:
                self.evecs_p
                self.evecs_t
            except AttributeError:
                self.diagonalise_p()
                self.diagonalise_t()

        self.probe_0_eb = self.evecs_p[0].T
        self.probe_1_eb = self.evecs_p[1].T
        self.probe_m_eb = 2**-0.5 * (self.probe_0_eb - self.probe_1_eb)
        self.probe_p_eb = 2**-0.5 * (self.probe_0_eb + self.probe_1_eb)

        self.target_0_eb = self.evecs_t[0].T
        self.target_1_eb = self.evecs_t[1].T
        self.target_m_eb = 2**-0.5 * (self.target_0_eb - self.target_1_eb)
        self.target_p_eb = 2**-0.5 * (self.target_0_eb + self.target_1_eb)
        
        self.target_0_cb = sp.sparse.csr_matrix((2 * self.ncut + 1,1))
        self.target_0_cb[self.ncut,0] = 1
        self.target_1_cb = sp.sparse.csr_matrix((2 * self.ncut + 1,1))
        self.target_1_cb[self.ncut + 1,0] = 1

    def init_prod_states(self, update=False):
        if update:
            self.init_probe_states(update=True)
        else:
            try:
                self.probe_0
            except AttributeError:
                self.init_probe_states()

        self.eb_eb_00 = sp.sparse.kron(self.probe_0_eb, self.target_0_eb)
        self.eb_eb_01 = sp.sparse.kron(self.probe_0_eb, self.target_1_eb)
        self.eb_eb_10 = sp.sparse.kron(self.probe_1_eb, self.target_0_eb)
        self.eb_eb_11 = sp.sparse.kron(self.probe_1_eb, self.target_1_eb)

        self.eb_cb_00 = sp.sparse.kron(self.probe_0_eb, self.target_0_cb)
        self.eb_cb_01 = sp.sparse.kron(self.probe_0_eb, self.target_1_cb)
        self.eb_cb_10 = sp.sparse.kron(self.probe_1_eb, self.target_0_cb)
        self.eb_cb_11 = sp.sparse.kron(self.probe_1_eb, self.target_1_cb)
        self.eb_cb_p0 = sp.sparse.kron(self.probe_p_eb, self.target_0_cb)
        self.eb_cb_p1 = sp.sparse.kron(self.probe_p_eb, self.target_1_cb)
        self.eb_cb_m0 = sp.sparse.kron(self.probe_m_eb, self.target_0_cb)
        self.eb_cb_m1 = sp.sparse.kron(self.probe_m_eb, self.target_1_cb)
    
    def calc_delta_p(self, update=False):
        if update:
            self.get_H_pt()
            self.init_prod_states(update=True)
        else:
            try:
                self.H_pt
            except AttributeError:
                self.get_H_pt()

            try:
                self.ket_p1
            except AttributeError:
                self.init_prod_states()

        delta_p = (self.hc(self.eb_eb_10).dot(self.H_pt.dot(self.eb_eb_10)) - self.hc(self.eb_eb_00).dot(self.H_pt.dot(self.eb_eb_00))).toarray()[0][0]
        
        return delta_p
    
    def calc_delta_t(self, update=False):
        if update:
            self.get_H_pt()
            self.init_prod_states(update=True)
        else:
            try:
                self.H_pt
            except AttributeError:
                self.get_H_pt()

            try:
                self.ket_p1
            except AttributeError:
                self.init_prod_states()

        delta_t = (self.hc(self.eb_eb_01).dot(self.H_pt.dot(self.eb_eb_01)) - self.hc(self.eb_eb_00).dot(self.H_pt.dot(self.eb_eb_00))).toarray()[0][0]

        return delta_t
    
    def calc_g_parr(self, update=False):
        if update:
            self.get_H_pt()
            self.init_prod_states(update=True)
        else:
            try:
                self.H_pt
            except AttributeError:
                self.get_H_pt()

            try:
                self.eb_cb_p1
            except AttributeError:
                self.init_prod_states()
        
        g_parr = self.hc(self.eb_cb_p1).dot(self.H_pt.dot(self.eb_cb_m1)).toarray()[0][0] - self.hc(self.eb_cb_p0).dot(self.H_pt.dot(self.eb_cb_m0)).toarray()[0][0]

        return g_parr
    
    def calc_g_perp(self, update=False):
        if update:
            self.get_H_pt()
            self.init_prod_states(update=True)
        else:
            try:
                self.H_pt
            except AttributeError:
                self.get_H_pt()

            try:
                self.eb_cb_11
            except AttributeError:
                self.init_prod_states()
        
        g_perp = self.hc(self.eb_cb_11).dot(self.H_pt.dot(self.eb_cb_01)).toarray()[0][0] - self.hc(self.eb_cb_10).dot(self.H_pt.dot(self.eb_cb_00)).toarray()[0][0]
        
        return g_perp