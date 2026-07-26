import unittest
import numpy as np

import sys
sys.path.append('C:/Users/charl/My Drive/Documents/Python3/MyPackages/liftingline')
import liftingline as ll


class TestParameters(unittest.TestCase):
    def test_Cl_Cd_relation(self):
        chord = .1
        combos = [[1.5, 20, 1.02],
                  [.5, 20, 1.02],
                  [1.5, 50, 1.40],
                  [5, 30, .9],]
        for combo in combos:
            span, v_inf, rho = combo
            AR = span/chord
            S = span*chord
            spans = np.array([0, span/2])
            chords = np.array([chord, chord])
            alphas = np.array([3, 1])
            wingshape = ll.WingShape(spans, chords, alphas)
            self.lls = ll.LiftingLineSolver(wingshape, v_inf, rho = rho)

            L, D = self.lls.get_total_forces()
            Cl = L/(.5*rho*v_inf**2*S)
            Cd = D/(.5*rho*v_inf**2*S)
            e_theory = Cl**2/(np.pi*AR*Cd)
            e_solver = self.lls.get_oswald()

            self.assertAlmostEqual(e_solver, e_theory, delta=1e-3)

class TestBasicWing(unittest.TestCase):
    def setUp(self):
        spans = np.array([0., 1., 2.])
        chords = np.array([.2, .15, .1])
        alphas = np.array([5, 5, 5])
        ail_spans = np.array([1.2, 1.6, 2.])
        wingshape = ll.WingShape(spans, chords, alphas, ail_spans)
        v_inf = 50
        self.lls = ll.LiftingLineSolver(wingshape, v_inf, nr_of_coefs = 10)

    def test_derivs(self):
        rate_combos = [[0, 0],
                       [-1.3, -.7],
                       [0, 0],
                       [-1.7, .5]]

        control_combos = [[0,0],
                          [0,0],
                          [.2,-.3],
                          [-.4,.1]]

        for i in range(len(rate_combos)):
            with self.subTest(i=i):
                rate_combo = rate_combos[i]
                control_combo = control_combos[i]
                self.eval_stab_derivs(rate_combo, control_combo)
                self.eval_cont_derivs(rate_combo, control_combo)
                
    def eval_stab_derivs(self, rate_combo, control_combo):
        q = .01
        margin = .0001
        
        p, r = rate_combo
        self.lls.set_rates(p, r)
        self.lls.set_control(control_combo)
        stab_derivs = self.lls.get_stability_derivs()
        Mx, Mz = self.lls.get_total_moments()

        d1, d2 = control_combo
        id_str = f"p:{p:.2f} r:{r:.2f} d1:{d1:.2f} d2:{d2:.2f}"

        # test p dependence
        self.lls.set_rates(p+q, r)
        Mx_p, Mz_p = self.lls.get_total_moments()
        dMxdp_discr = (Mx_p - Mx)/q
        dMzdp_discr = (Mz_p - Mz)/q

        dMxdp = stab_derivs[0,0]
        dMzdp = stab_derivs[1,0]

        tol_xp = abs(dMxdp*margin)
        tol_zp = abs(dMzdp*margin)
                

        self.assertAlmostEqual(dMxdp, dMxdp_discr, delta=tol_xp, msg = id_str)
        self.assertAlmostEqual(dMzdp, dMzdp_discr, delta=tol_zp, msg = id_str)

        # test r dependence
        self.lls.set_rates(p, r+q)
        Mx_r, Mz_r = self.lls.get_total_moments()
        dMxdr_discr = (Mx_r - Mx)/q
        dMzdr_discr = (Mz_r - Mz)/q

        dMxdr = stab_derivs[0,1]
        dMzdr = stab_derivs[1,1]

        tol_xr = abs(dMxdr*margin)
        tol_zr = abs(dMzdr*margin)

        self.assertAlmostEqual(dMxdr, dMxdr_discr, delta=tol_xr, msg = id_str)
        self.assertAlmostEqual(dMzdr, dMzdr_discr, delta=tol_zr, msg = id_str)

    def eval_cont_derivs(self, rate_combo, control_combo):
        q = .01
        margin = .0001

        p, r = rate_combo
        d1, d2 = control_combo
        self.lls.set_rates(p, r)
        self.lls.set_control(control_combo)
        cont_derivs = self.lls.get_control_derivs()
        Mx, Mz = self.lls.get_total_moments()
        
        id_str = f"p:{p:.2f} r:{r:.2f} d1:{d1:.2f} d2:{d2:.2f}"

        for i, cont_input in enumerate([[d1+q, d2],[d1, d2+q]]):
            self.lls.set_control(cont_input)
            Mx_k, Mz_k = self.lls.get_total_moments()
            dMxdk_discr = (Mx_k - Mx)/q
            dMzdk_discr = (Mz_k - Mz)/q

            dMxdk_lower = dMxdk_discr*(1-margin)
            dMxdk_upper = dMxdk_discr*(1+margin)

            dMxdk = cont_derivs[0,i]
            dMzdk = cont_derivs[1,i]

            tol_xk = abs(dMxdk*margin)
            tol_zk = abs(dMzdk*margin)
        
            self.assertAlmostEqual(dMxdk, dMxdk_discr, delta=tol_xk, msg = id_str)
            self.assertAlmostEqual(dMzdk, dMzdk_discr, delta=tol_zk, msg = id_str)

            
    def test_control(self):
        delta_combos = [[.1 , 0],
                        [0  , .1],
                        [.1 , .2],
                        [-.1, 0],
                        [0  , -.2],
                        [-.1, .2],
                        [-.2, .1]]
        L_base, D_base = self.lls.get_total_forces()
        for i, delta_combo in enumerate(delta_combos):
            with self.subTest(i=i): 
                self.lls.set_control_steady_roll(delta_combo)
                L, D = self.lls.get_total_forces()
                Mx, Mz = self.lls.get_total_moments()
                self.assertAlmostEqual(L_base, L, msg = f'combo nr = {i}', delta=1e-3)
                self.assertAlmostEqual(Mx, 0,  msg = f'combo nr = {i}', delta=1e-3)

        
if __name__ == "__main__":
    unittest.main()


# Back up

def off_test_A_coefs(self):
    q = .001
    dAdp, dAdr = self.lls.get_derivs()
    A = self.lls.A.copy()

    self.lls.set_rates(q, 0)
    A_p = self.lls.A.copy()
    
    self.lls.set_rates(0, q)
    A_r = self.lls.A.copy()

    dAdp_discr = (A_p - A)/q
    dAdr_discr = (A_r - A)/q

    error_p = np.sum((dAdp - dAdp_discr)**2)
    error_r = np.sum((dAdr - dAdr_discr)**2)
    
    self.assertAlmostEqual(error_p, 0, delta=dAdp.max()*.0005)
    self.assertAlmostEqual(error_r, 0, delta=dAdr.max()*.0005)

def off_test_gamma(self):
    q = .01
    margin = .005
    dgammadp, dgammadr = self.lls.get_derivs()
    gamma = self.lls.gamma.copy()

    self.lls.set_rates(q, 0)
    gamma_p = self.lls.gamma.copy()
    
    self.lls.set_rates(0, q)
    gamma_r = self.lls.gamma.copy()

    dgammadp_discr = (gamma_p - gamma)/q
    dgammadr_discr = (gamma_r - gamma)/q

    error_p = np.sum((dgammadp - dgammadp_discr)**2)
    error_r = np.sum((dgammadr - dgammadr_discr)**2)

    
    self.assertAlmostEqual(error_p, 0, delta=1e-3)
    self.assertAlmostEqual(error_r, 0, delta=1e-3)

def off_test_alpha(self):
    q = .01
    margin = .005
    dalphadp, dalphadr = self.lls.get_derivs()
    
    alpha = self.lls.alpha_geo[1:-1] + self.lls.alpha_pr_over_p[1:-1]*self.lls.p

    self.lls.set_rates(q, 0)
    alpha_p = self.lls.alpha_geo[1:-1] + self.lls.alpha_pr_over_p[1:-1]*self.lls.p
    
    self.lls.set_rates(0, q)
    alpha_r = self.lls.alpha_geo[1:-1] + self.lls.alpha_pr_over_p[1:-1]*self.lls.p

    dalphadp_discr = (alpha_p - alpha)/q
    dalphadr_discr = (alpha_r - alpha)/q

    error_p = np.sum((dalphadp - dalphadp_discr)**2)
    error_r = np.sum((dalphadr - dalphadr_discr)**2)
    
    self.assertAlmostEqual(error_p, 0, delta=1e-3)
    self.assertAlmostEqual(error_r, 0, delta=1e-3)

def off_test_M(self):
    q = .01
    margin = .005
    dMdr = self.lls.get_derivs()
    
    M = np.linalg.inv(self.lls.M_inv)
    
    self.lls.set_rates(0, q)
    M_r = np.linalg.inv(self.lls.M_inv)

    dMdr_discr = (M_r - M)/q

    error_r = np.sum((dMdr - dMdr_discr)**2)
    
    self.assertAlmostEqual(error_r, 0, delta=1e-3)
