import numpy as np
import scipy as sp
import scipy.sparse.linalg
import scipy.sparse as sparse

class DCCircuit:
    def __init__(self, Ejp=153e9, Ejt=21.3e9, Cjp=12e-15, Cjt=36e-15, Cc=3.1e-15, Ct=18e-15, alpha=0.25, ng=0.12, flux=0.5, ncut=2):
        self.h = 6.626e-34
        self.hbar = 1.055e-34
        self.e_charge = -1.60218e-19
        
        self.Ejp = Ejp * self.h                 # Josephson energy of probe
        self.Ejt = Ejt * self.h                 # Josephson energy of target
        self.Cjp = Cjp                          # Josephson capacitance of probe
        self.Cjt = Cjt                          # Josephson capacitance of target
        self.Ct = Ct                            # Target Capacitor
        self.Cc = Cc                            # Coupling capacitance
        self.ng = ng                            # Reduced gate charge
        self.flux = flux                        # Flux through qubit
        self.ncut = ncut                        # Cut-off threshold for number basis
        self.alpha = alpha
        
        self.init_operators()
        
    def _print_params(self):
        print(f'Ejp:    {self.Ejp * 1e-9 / h} GHz')
        print(f'Ejt:    {self.Ejt * 1e-9 / h} GHz')
        print(f'Cjp:    {self.Cjp * 1e15} fF')
        print(f'Cjt:    {self.Cjt * 1e15} fF')
        print(f'Ct:    {self.Ct * 1e15} fF')
        print(f'Cc:    {self.Cc * 1e15} fF')
        print(f'alpha: {self.alpha}')
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

    def mod_squared(self, val):
        return np.real(val * np.conjugate(val))
        
    def get_kin_pt(self):
        self.init_operators()
        
        C_mat = [
            [(1 + self.alpha) * self.Cjp, 0, -self.alpha * self.Cjp, 0],
            [0, (1 + self.alpha) * self.Cjp, -self.alpha * self.Cjp, 0],
            [-self.alpha * self.Cjp, -self.alpha * self.Cjp, 2 * self.alpha * self.Cjp + self.Cc, -self.Cc],
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
        pot_pt += -self.Ejp * 0.5 * self.alpha * self.tensor4(self.e_iphi_op_cb.T, self.I_cb, self.e_iphi_op_cb, self.I_cb)
        pot_pt += -self.Ejp * 0.5 * self.alpha * self.tensor4(self.e_iphi_op_cb, self.I_cb, self.e_iphi_op_cb.T, self.I_cb)
        pot_pt += -self.Ejp * 0.5 * self.alpha * np.exp(2j * np.pi * self.flux) * self.tensor4(self.I_cb, self.e_iphi_op_cb, self.e_iphi_op_cb.T, self.I_cb)
        pot_pt += -self.Ejp * 0.5 * self.alpha * np.exp(-2j * np.pi * self.flux) * self.tensor4(self.I_cb, self.e_iphi_op_cb.T, self.e_iphi_op_cb, self.I_cb)
        pot_pt += self.Ejp * 2 * (1 + self.alpha) * self.tensor4(self.I_cb, self.I_cb, self.I_cb, self.I_cb)
        
        pot_pt += self.Ejt * self.tensor4(self.I_cb, self.I_cb, self.I_cb, self.I_cb)        
        pot_pt += -self.Ejt * self.tensor4(self.I_cb, self.I_cb, self.I_cb, self.e_iphi_op_cb + self.e_iphi_op_cb.T)
        
        return pot_pt
        
    def get_kin_p(self):
        self.init_operators()
        
        C_mat = [
            [(1 + self.alpha) * self.Cjp, 0, -self.alpha * self.Cjp],
            [0, (1 + self.alpha) * self.Cjp, -self.alpha * self.Cjp],
            [-self.alpha * self.Cjp, -self.alpha * self.Cjp, 2 * self.alpha * self.Cjp + self.Cc]
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
        pot_p += -self.Ejp * 0.5 * self.alpha * self.tensor3(self.e_iphi_op_cb.T, self.I_cb, self.e_iphi_op_cb)
        pot_p += -self.Ejp * 0.5 * self.alpha * self.tensor3(self.e_iphi_op_cb, self.I_cb, self.e_iphi_op_cb.T)
        pot_p += -self.Ejp * 0.5 * self.alpha * np.exp(2j * np.pi * self.flux) * self.tensor3(self.I_cb, self.e_iphi_op_cb, self.e_iphi_op_cb.T)
        pot_p += -self.Ejp * 0.5 * self.alpha * np.exp(-2j * np.pi * self.flux) * self.tensor3(self.I_cb, self.e_iphi_op_cb.T, self.e_iphi_op_cb)
        pot_p += -self.Ejp * 0.5 * self.tensor3(self.I_cb, self.e_iphi_op_cb + self.e_iphi_op_cb.T, self.I_cb)
        pot_p += self.Ejp * 2 * (1 + self.alpha) * self.tensor3(self.I_cb, self.I_cb, self.I_cb)
        
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
    
    def init_qubit_states(self, update=False):
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
        self.probe_p_eb = 2**-0.5 * (self.probe_0_eb + self.probe_1_eb)
        self.probe_m_eb = 2**-0.5 * (self.probe_0_eb - self.probe_1_eb)
        self.probe_ip_eb = 2**-0.5 * (self.probe_0_eb + 1j * self.probe_1_eb)
        self.probe_im_eb = 2**-0.5 * (self.probe_0_eb - 1j * self.probe_1_eb)

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
            self.init_qubit_states(update=True)
        else:
            try:
                self.probe_0
            except AttributeError:
                self.init_qubit_states()

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
    
    def calc_omega01(self, update=False):
        if update:
            self.get_H_p()
            self.diagonalise_p()
        else:
            try:
                self.H_p
            except AttributeError:
                self.get_H_p()
            try:
                self.evals_p
            except AttributeError:
                self.diagonalise_p()

        return (self.evals_p[1] - self.evals_p[0]) / self.h
    
    def calc_T1(self, ng0=0.25, flux0=0.5, diff_param=0.001, update=True):
        self.ng = ng0
        self.flux = flux0
        self.init_qubit_states(update=True)
        omega01 = self.calc_omega01()

        self.ng = ng0 + 0.5 * diff_param
        self.init_qubit_states(update=True)
        D_high = self.hc(self.probe_1_eb).dot(self.get_H_p().dot(self.probe_0_eb)).toarray()[0][0]

        self.ng = ng0 - 0.5 * diff_param
        self.init_qubit_states(update=True)
        D_low = self.hc(self.probe_1_eb).dot(self.get_H_p().dot(self.probe_0_eb)).toarray()[0][0]

        dH = (D_high - D_low) / diff_param
        print(dH / self.h)
        print((5.2)**2 * omega01)

        gamma_1 = (2 / self.h**2) * self.mod_squared(dH) * (5.2)**2 * omega01

        self.ng = ng0

        self.flux = flux0 + 0.5 * diff_param
        self.init_qubit_states(update=True)
        D_high = self.hc(self.probe_1_eb).dot(self.get_H_p().dot(self.probe_0_eb)).toarray()[0][0]

        self.flux = flux0 - 0.5 * diff_param
        self.init_qubit_states(update=True)
        D_low = self.hc(self.probe_1_eb).dot(self.get_H_p().dot(self.probe_0_eb)).toarray()[0][0]

        dH = (D_high - D_low) / diff_param
        print(dH / self.h)

        print(gamma_1)
        # gamma_1 += (2 / self.h**2) * self.mod_squared(dH) * (5.2)**2 * omega01
        # print(gamma_1)

        return 1 / gamma_1
    
    def calc_Tphi(self):
        self.calc_D()

        A = 1e-4
        omega_low = 1
        omega_high = 3e9
        t_exp = 10e-6

        gamma_phi = 2 * A**2 * self.D_omega01**2 * np.abs(np.log(omega_low * t_exp))
        gamma_phi += 2 * A**4 * self.D2_omega01**2 * (np.log(omega_high/omega_low)**2 + np.log(omega_low * t_exp)**2)
        gamma_phi = np.sqrt(gamma_phi)

        return 1 / gamma_phi
    
    def calc_T2(self):

        T1 = self.calc_T1()
        Tphi = self.calc_T2()

        gamma_t2 = (1 / (2 * T1)) + (1 / Tphi)

        return 1 / gamma_t2

    def calc_D(self, ng0=0.5, diff_param=0.01):
        self.ng = ng0
        self.init_qubit_states(update=True)
        H_p = self.get_H_p()

        self.ng = ng0 + 0.5 * diff_param
        evals_high, _ = self.diagonalise_p(update=True)
        omega01_high = (evals_high[1] - evals_high[0]) / self.hbar

        self.ng = ng0 - 0.5 * diff_param
        evals_low, _ = self.diagonalise_p(update=True)
        omega01_low = (evals_low[1] - evals_low[0]) / self.hbar
        
        self.D_omega01 = np.abs(omega01_high - omega01_low) / diff_param

        self.ng = ng0
        evals, _ = self.diagonalise_p(update=True)
        omega01 = (evals[1] - evals[0]) / self.hbar

        self.ng = ng0 + diff_param
        evals_high, _ = self.diagonalise_p(update=True)
        omega01_high = (evals_high[1] - evals_high[0]) / self.hbar

        self.ng = ng0 - diff_param
        evals_low, _ = self.diagonalise_p(update=True)
        omega01_low = (evals_low[1] - evals_low[0]) / self.hbar
        
        self.D2_omega01 = np.abs(omega01_high - 2 * omega01 + omega01_low) / diff_param**2

        self.ng = ng0
        self.diagonalise_p(update=True)

        self.ng = ng0 + 0.5 * diff_param
        H_p_high = self.get_H_p()
        H_z_high = self.hc(self.probe_1_eb).dot(H_p_high.dot(self.probe_1_eb)).toarray()[0][0]

        self.ng = ng0 - 0.5 * diff_param
        H_p_low = self.get_H_p()
        H_z_low = self.hc(self.probe_1_eb).dot(H_p_low.dot(self.probe_1_eb)).toarray()[0][0]

        self.D_z = np.sqrt(self.mod_squared((H_z_high - H_z_low) / diff_param)) / self.h

        self.ng = ng0
        H_p = self.get_H_p()
        H_z = self.hc(self.probe_1_eb).dot(H_p.dot(self.probe_1_eb)).toarray()[0][0]

        self.ng = ng0 + diff_param
        H_p = self.get_H_p()
        H_z_high = self.hc(self.probe_1_eb).dot(H_p.dot(self.probe_1_eb)).toarray()[0][0]

        self.ng = ng0 - diff_param
        H_p = self.get_H_p()
        H_z_low = self.hc(self.probe_1_eb).dot(H_p.dot(self.probe_1_eb)).toarray()[0][0]

        self.D2_z = np.sqrt(self.mod_squared((H_z_high - 2 * H_z + H_z_low) / diff_param**2)) / self.h**2

        self.ng = ng0 + 0.5 * diff_param
        H_p = self.get_H_p()
        H_x_high = self.hc(self.probe_p_eb).dot(H_p.dot(self.probe_p_eb)).toarray()[0][0]
        H_y_high = self.hc(self.probe_ip_eb).dot(H_p.dot(self.probe_ip_eb)).toarray()[0][0]

        self.ng = ng0 - 0.5 * diff_param
        H_p = self.get_H_p()
        H_x_low = self.hc(self.probe_p_eb).dot(H_p.dot(self.probe_p_eb)).toarray()[0][0]
        H_y_low = self.hc(self.probe_ip_eb).dot(H_p.dot(self.probe_ip_eb)).toarray()[0][0]

        self.D_perp = np.sqrt(self.mod_squared((H_x_high - H_x_low) / diff_param) + self.mod_squared((H_y_high - H_y_low) / diff_param)) / self.h

class DCCircuitAlphaDisp:
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