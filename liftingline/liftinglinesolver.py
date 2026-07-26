import numpy as np


class LiftingLineSolver:
    def __init__(self, wingshape, v_inf, rho=1.225, nr_of_coefs=100):
        self.wingshape = wingshape
        self.v_inf = v_inf
        self.rho = rho
        self.nr_of_coefs = nr_of_coefs
        self.n_llt_lst = np.arange(nr_of_coefs-2)+1

        self.theta = np.linspace(0,np.pi,self.nr_of_coefs)
        self.cos_theta = np.cos(self.theta)
        self.y = self._theta_to_y(self.theta)
        self.chords = wingshape.chord(self.y)

        self.p = 0
        self.r = 0

        self.deltas = np.zeros(wingshape.nr_of_controls)

        self._solve_lift()

    def _solve_lift(self):
        self.v_theta = self.v_inf + self.r*self.wingshape.span*self.cos_theta/2
        self.alpha_pr_over_p = - self.wingshape.span * self.cos_theta / (2*self.v_theta)
        self.alpha_geo = self._make_alpha_geo()
        self.M_inv = self._calc_M_inv()
        self.A = self.M_inv@ (self.alpha_geo[1:-1] + self.alpha_pr_over_p[1:-1]*self.p)
        self.gamma = self._calc_gamma(self.A)
        self.L = self.rho*self.v_theta*self.gamma
        self.alpha_eff = self.gamma / (np.pi*self.v_theta*self.chords)
        self.D = self.L * (self.alpha_geo - self.alpha_eff)

    def _theta_to_y(self, theta):
        b = self.wingshape.span
        return -b/2 * np.cos(theta)

    def _calc_M_inv(self):
        n_llt_lst_loc = self.n_llt_lst[:, np.newaxis]
        
        s1 = np.sin(n_llt_lst_loc * self.theta[1:-1])
        s3 = s1 / np.sin(self.theta[1:-1])
        s2 = n_llt_lst_loc * s3
        M_T = (2 * self.wingshape.span / (np.pi * self.chords[1:-1])) * s1 + s2 - self.r*self.alpha_pr_over_p[1:-1]*s3 
        M = M_T.T
        return np.linalg.inv(M)

    def _calc_gamma(self, A):
        b = self.wingshape.span
        n_llt_lst_loc = self.n_llt_lst[:, np.newaxis]
        sin_terms = np.sin(n_llt_lst_loc * self.theta)
        dgammadr = 2 * b * self.v_theta * (sin_terms.T @ A)
        return dgammadr

    def _make_alpha_geo(self):
        alpha_controls = np.zeros(self.nr_of_coefs)
        for delta_nr in range(self.wingshape.nr_of_controls):
            alpha_controls += self.deltas[delta_nr] * self.wingshape.alpha_control(self.y, delta_nr)
        alpha_wing = self.wingshape.alpha_geo(self.y)
        return alpha_wing + alpha_controls
        
    def get_stability_derivs(self):
        b = self.wingshape.span
        dMdr = self._calc_dMdr()
        dalphadp = self.alpha_pr_over_p[1:-1]
        dalphadr = self.alpha_pr_over_p[1:-1]**2 *self.p
        
        dAdp = self.M_inv@ dalphadp
        dAdr = -self.M_inv @ dMdr @ self.M_inv @ (self.alpha_geo[1:-1] + self.alpha_pr_over_p[1:-1]*self.p) + self.M_inv @ dalphadr

        dgammadp = self._calc_gamma(dAdp)
        dgammadr = self._calc_dgammadr(dAdr)
        
        dLdp = self.rho*self.v_theta*dgammadp
        dLdr = self.rho*self.v_theta*dgammadr + self.rho*self.gamma*b*self.cos_theta/2
        
        dalpha_effdp = dgammadp / (np.pi*self.v_theta*self.chords)
        dalpha_effdr = (dgammadr + self.alpha_pr_over_p * self.gamma) / (np.pi*self.v_theta*self.chords)
        
        dDdp = dLdp * (self.alpha_geo - self.alpha_eff) - self.L * dalpha_effdp
        dDdr = dLdr * (self.alpha_geo - self.alpha_eff) - self.L * dalpha_effdr

        dMxdp = self._distribution_to_moment(dLdp)
        dMxdr = self._distribution_to_moment(dLdr)
        dMzdp = - self._distribution_to_moment(dDdp)
        dMzdr = - self._distribution_to_moment(dDdr)
        return np.array([[dMxdp, dMxdr], [dMzdp, dMzdr]])

    def get_derivs(self):
        b = self.wingshape.span
        dMdr = self._calc_dMdr()
        dalphadp = self.alpha_pr_over_p[1:-1]
        dalphadr = self.alpha_pr_over_p[1:-1]**2 *self.p
        
        dAdp = self.M_inv@ dalphadp
        dAdr = -self.M_inv @ dMdr @ self.M_inv @ (self.alpha_geo[1:-1] + self.alpha_pr_over_p[1:-1]*self.p) + self.M_inv @ dalphadr

        dgammadp = self._calc_gamma(dAdp)
        dgammadr = self._calc_dgammadr(dAdr)
        
        dLdp = self.rho*self.v_theta*dgammadp
        dLdr = self.rho*self.v_theta*dgammadr + self.rho*self.gamma*b*self.cos_theta/2
        
        dalpha_effdp = dgammadp / (np.pi*self.v_theta*self.chords)
        dalpha_effdr = (dgammadr + self.alpha_pr_over_p * self.gamma) / (np.pi*self.v_theta*self.chords)
        
        dDdp = dLdp * (self.alpha_geo - self.alpha_eff) - self.L * dalpha_effdp
        dDdr = dLdr * (self.alpha_geo - self.alpha_eff) - self.L * dalpha_effdr
        return dMdr

    def _calc_dMdr(self):
        n_llt_lst_loc = self.n_llt_lst[:, np.newaxis]
        s1 = np.sin(n_llt_lst_loc * self.theta[1:-1])
        s3 = s1 / np.sin(self.theta[1:-1])
        dMdr_T = - self.alpha_pr_over_p[1:-1] * self.v_inf / self.v_theta[1:-1] * s3 
        return dMdr_T.T

    def _calc_dgammadr(self, dAdr):
        b = self.wingshape.span
        n_llt_lst_loc = self.n_llt_lst[:, np.newaxis]
        sin_terms = np.sin(n_llt_lst_loc * self.theta)
        dgammadr = 2 * b * self.v_theta * (sin_terms.T @ dAdr) - self.alpha_pr_over_p * self.gamma
        return dgammadr

    def _distribution_to_moment(self, distribution):
        b = self.wingshape.span
        integrand = np.sin(2*self.theta) * distribution
        moment = b**2/8 * np.trapezoid(integrand, self.theta)
        return moment

    def get_control_derivs(self):
        control_derivs = np.zeros((2, self.wingshape.nr_of_controls))

        for delta_nr in range(self.wingshape.nr_of_controls):
            b = self.wingshape.span
            alpha_control = self.wingshape.alpha_control(self.y, delta_nr)
            dAdk = self.M_inv@ alpha_control[1:-1]

            dgammadk = self._calc_gamma(dAdk)
            dLdk = self.rho*self.v_theta*dgammadk
            dalpha_effdk = dgammadk / (np.pi*self.v_theta*self.chords)
            dDdk = dLdk * (self.alpha_geo - self.alpha_eff) + self.L * (alpha_control - dalpha_effdk)

            control_derivs[0,delta_nr] = self._distribution_to_moment(dLdk)
            control_derivs[1,delta_nr] = -self._distribution_to_moment(dDdk)
        return control_derivs

    def get_oswald(self):
        A = self.A
        A1 = A[0]  
        higher_order_terms = np.sum((np.arange(2, len(A) + 1) * A[1:]**2))
        e = 1 / (1 + (higher_order_terms / A1**2))
        return e

    def get_total_forces(self):
        L_total = np.trapezoid(self.L, self.y)
        D_total = np.trapezoid(self.D, self.y)
        return L_total, D_total

    def get_total_moments(self):
        Mx = self._distribution_to_moment(self.L)
        Mz = -self._distribution_to_moment(self.D)
        return Mx, Mz

    def set_control(self, deltas):
        self.deltas = np.array(deltas)
        self._solve_lift()

    def set_rates(self, p, r):
        self.p = p
        self.r = r
        self._solve_lift()

    def set_control_steady_roll(self, deltas):
        self.set_rates(0, 0)
        self.set_control(deltas)
        Mx, Mz = self.get_total_moments()
        stab_derivs = self.get_stability_derivs()
        dMxdp = stab_derivs[0,0]
        self.p = -Mx/dMxdp
        self._solve_lift()
