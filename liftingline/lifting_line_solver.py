import numpy as np
from liftingline.lifting_line_solution import LiftingLineSolution


class LiftingLineSolver:
    def __init__(self, wingshape, nr_of_coefs: int = 100):
        self.wingshape = wingshape
        self.nr_of_coefs = nr_of_coefs
        self.n_llt_lst = np.arange(nr_of_coefs - 2) + 1

        # Precompute static grid and geometry
        self.theta = np.linspace(0, np.pi, self.nr_of_coefs)
        self.cos_theta = np.cos(self.theta)
        self.y = -self.wingshape.span / 2 * self.cos_theta
        self.chords = wingshape.chord(self.y)

    def solve(
        self,
        v_inf: float,
        rho: float = 1.225,
        p: float = 0.0,
        r: float = 0.0,
        deltas: list[float] | np.ndarray = None,
        eval_derivs: bool = True,
        ) -> LiftingLineSolution:
        
        b = self.wingshape.span

        if deltas is None:
            deltas = np.zeros(self.wingshape.nr_of_controls)
        else:
            deltas = np.asarray(deltas, dtype=float)

        # 1. Effective flow velocity & angles
        v_theta = v_inf + r * b * self.cos_theta / 2.0
        alpha_pr_over_p = -b * self.cos_theta / (2.0 * v_theta)

        # 2. Geometric angle of attack (baseline + control deflections)
        alpha_controls = np.zeros(self.nr_of_coefs)
        for delta_nr in range(self.wingshape.nr_of_controls):
            alpha_controls += deltas[
                delta_nr
            ] * self.wingshape.alpha_control(self.y, delta_nr)
        alpha_geo = self.wingshape.alpha_geo(self.y) + alpha_controls

        # 3. System Matrix Assembly & Solve
        M_inv = self._calc_M_inv(v_theta, alpha_pr_over_p, r)
        A = M_inv @ (alpha_geo[1:-1] + alpha_pr_over_p[1:-1] * p)

        # 4. Circulation & Forces
        gamma = self._calc_gamma(A, v_theta)
        L = rho * v_theta * gamma
        alpha_eff = gamma / (np.pi * v_theta * self.chords)
        D = L * (alpha_geo - alpha_eff)

        # 5. Integrated Forces & Moments
        L_total = np.trapezoid(L, self.y)
        D_total = np.trapezoid(D, self.y)
        Mx = self._distribution_to_moment(L)
        Mz = -self._distribution_to_moment(D)

        # 6. Stability & Control Derivatives
        if eval_derivs:
            stab_derivs = self._calc_stability_derivs(
                rho,
                v_inf,
                p,
                r,
                v_theta,
                alpha_pr_over_p,
                alpha_geo,
                alpha_eff,
                gamma,
                L,
                M_inv,
                A,
            )
            ctrl_derivs = self._calc_control_derivs(
                rho, v_theta, alpha_geo, alpha_eff, L, M_inv
            )
        else:
            stab_derivs = np.zeros((2, 2))
            ctrl_derivs = np.zeros((2, self.wingshape.nr_of_controls))

        return LiftingLineSolution(
            v_inf=v_inf,
            p=p,
            r=r,
            deltas=deltas,
            y=self.y,
            theta=self.theta,
            chords=self.chords,
            alpha_geo=alpha_geo,
            alpha_eff=alpha_eff,
            gamma=gamma,
            L=L,
            D=D,
            fourier_coeffs=A,
            L_total=L_total,
            D_total=D_total,
            Mx=Mx,
            Mz=Mz,
            stability_derivs=stab_derivs,
            control_derivs=ctrl_derivs,
        )

    # --- Matrix Math Helpers ---

    def _calc_M_inv(self, v_theta, alpha_pr_over_p, r):
        n_llt = self.n_llt_lst[:, np.newaxis]
        s1 = np.sin(n_llt * self.theta[1:-1])
        s3 = s1 / np.sin(self.theta[1:-1])
        s2 = n_llt * s3
        M_T = (
            (2 * self.wingshape.span / (np.pi * self.chords[1:-1])) * s1
            + s2
            - r * alpha_pr_over_p[1:-1] * s3
        )
        return np.linalg.inv(M_T.T)

    def _calc_gamma(self, A, v_theta):
        n_llt = self.n_llt_lst[:, np.newaxis]
        sin_terms = np.sin(n_llt * self.theta)
        return 2 * self.wingshape.span * v_theta * (sin_terms.T @ A)

    def _distribution_to_moment(self, distribution):
        integrand = np.sin(2 * self.theta) * distribution
        return (self.wingshape.span**2 / 8) * np.trapezoid(
            integrand, self.theta
        )

    # --- Derivative Calculations ---

    def _calc_stability_derivs(
        self,
        rho,
        v_inf,
        p,
        r,
        v_theta,
        alpha_pr_over_p,
        alpha_geo,
        alpha_eff,
        gamma,
        L,
        M_inv,
        A,
    ):
        b = self.wingshape.span

        # Derivative of system matrix wrt r rate
        n_llt = self.n_llt_lst[:, np.newaxis]
        s1 = np.sin(n_llt * self.theta[1:-1])
        s3 = s1 / np.sin(self.theta[1:-1])
        dMdr = (-alpha_pr_over_p[1:-1] * v_inf / v_theta[1:-1] * s3).T

        dalphadp = alpha_pr_over_p[1:-1]
        dalphadr = (alpha_pr_over_p[1:-1] ** 2) * p

        dAdp = M_inv @ dalphadp
        dAdr = -M_inv @ dMdr @ M_inv @ (
            alpha_geo[1:-1] + alpha_pr_over_p[1:-1] * p
        ) + M_inv @ dalphadr

        dgammadp = self._calc_gamma(dAdp, v_theta)
        sin_terms = np.sin(n_llt * self.theta)
        dgammadr = (
            2 * b * v_theta * (sin_terms.T @ dAdr) - alpha_pr_over_p * gamma
        )

        dLdp = rho * v_theta * dgammadp
        dLdr = (
            rho * v_theta * dgammadr
            + rho * gamma * b * self.cos_theta / 2.0
        )

        dalpha_effdp = dgammadp / (np.pi * v_theta * self.chords)
        dalpha_effdr = (dgammadr + alpha_pr_over_p * gamma) / (
            np.pi * v_theta * self.chords
        )

        dDdp = dLdp * (alpha_geo - alpha_eff) - L * dalpha_effdp
        dDdr = dLdr * (alpha_geo - alpha_eff) - L * dalpha_effdr

        dMxdp = self._distribution_to_moment(dLdp)
        dMxdr = self._distribution_to_moment(dLdr)
        dMzdp = -self._distribution_to_moment(dDdp)
        dMzdr = -self._distribution_to_moment(dDdr)

        return np.array([[dMxdp, dMxdr], [dMzdp, dMzdr]])

    def _calc_control_derivs(self, rho,v_theta, alpha_geo, alpha_eff, L, M_inv):
        nr_ctrls = self.wingshape.nr_of_controls
        control_derivs = np.zeros((2, nr_ctrls))

        for k in range(nr_ctrls):
            alpha_control = self.wingshape.alpha_control(self.y, k)
            dAdk = M_inv @ alpha_control[1:-1]

            dgammadk = self._calc_gamma(dAdk, v_theta)
            dLdk = rho * v_theta * dgammadk
            dalpha_effdk = dgammadk / (np.pi * v_theta * self.chords)
            dDdk = dLdk * (alpha_geo - alpha_eff) + L * (
                alpha_control - dalpha_effdk
            )

            control_derivs[0, k] = self._distribution_to_moment(dLdk)
            control_derivs[1, k] = -self._distribution_to_moment(dDdk)

        return control_derivs
