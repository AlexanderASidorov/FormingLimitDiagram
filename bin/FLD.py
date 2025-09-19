import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


class FLCPredictor:
    def __init__(self, A80, r0, r45, r90, t, A80min=None):
        """
        Инициализация модели прогнозирования FLC
        
        Параметры:
            A80 (float): общее удлинение при растяжении (%)
            r (float):   коэффициент нормальной анизотропии
            t (float):   толщина материала (мм)
            A80min (float): минимальное удлинение по разным направлениям (по умолчанию = A80)
        """
        self.A80 = A80
        self.r0 = r0
        self.r45 = r45
        self.r90 = r90
        self.r = (r0 + 2*r45 + r90) / 4
        self.t = t
        self.A80min = A80min if A80min is not None else A80

        # Расчёт точек FLC
        self._calculate_all_points()

    def _calculate_all_points(self):
        """Рассчитывает все 4 ключевые точки FLC"""
        self.TE = self.calculate_TE_point(self.A80, self.r, self.t)
        self.PS = self.calculate_PS_point(self.A80, self.t)
        self.IM = self.calculate_IM_point(self.A80, self.t)
        self.BI = self.calculate_BI_point(self.A80, self.t, self.A80min)

    def calculate_TE_point(self, A80, r, t):
        """
        Формулы (14)-(16) — деформации в точке TE (uniaxial tension necking point)
        """
        term_A = 0.0626 * A80 ** 0.567
        term_t = (t - 1) * (0.12 - 0.0024 * A80)
        SVL_t = term_A + term_t
        strain_ratio = 0.797 * r ** 0.701
        denominator = np.sqrt(1 + strain_ratio ** 2)

        eps3_TE = -SVL_t / denominator
        eps2_TE = -SVL_t * strain_ratio / denominator
        eps1_TE = (1 + strain_ratio) * SVL_t / denominator

        return {'eps2': eps2_TE, 'eps1': eps1_TE}

    def calculate_PS_point(self, A80, t):
        """
        Формула (20) — точка PS (plane strain point)
        """
        eps1_PS = 0.0084 * A80 + 0.0017 * A80 * (t - 1)
        return {'eps2': 0.0, 'eps1': eps1_PS}

    def calculate_BI_point(self, A80, t, A80min=None):
        """
        Формулы (21), (25), (26) — точка BI (biaxial stretch point)
        """
        A80min = A80min or A80

        eps1_BI_1mm = 0.005 * A80min + 0.25
        t_trans = 1.5 - (0.00215 * A80min) / (0.6 + 0.00285 * A80min)

        if t <= t_trans:
            eps1_BI = 0.00215 * A80min + 0.25 + 0.00285 * A80min * t
        else:
            eps1_BI = eps1_BI_1mm

        return {'eps2': eps1_BI, 'eps1': eps1_BI}

    def calculate_IM_point(self, A80, t):
        """
        Формулы (27), (31), (32) — точка IM (intermediate biaxial stretch point)
        """
        t_trans = 1.5 - (0.00215 * A80) / (0.6 + 0.00285 * A80)
        base_eps_IM1 = 0.0062 * A80 + 0.18

        if t <= t_trans:
            eps1_IM = 0.0062 * A80 + 0.18 + 0.0027 * A80 * (t - 1)
        else:
            eps1_IM = base_eps_IM1

        return {'eps2': 0.75 * eps1_IM, 'eps1': eps1_IM}

    def get_FLC_points(self):
        """Возвращает точки FLC как словарь"""
        return {
            'TE': (self.TE['eps2'], self.TE['eps1']),
            'PS': (self.PS['eps2'], self.PS['eps1']),
            'IM': (self.IM['eps2'], self.IM['eps1']),
            'BI': (self.BI['eps2'], self.BI['eps1'])
        }

    def round_down_to_nearest_tenth(self, x):
        """Округление вниз до ближайшего 0.1"""
        return np.floor(x * 10) / 10

    def round_up_to_nearest_tenth(self, x):
        """Округление вверх до ближайшего 0.1"""
        return np.ceil(x * 10) / 10

    def extrapolate_FLC_dynamically(self, x_vals, y_vals):
        """
        Динамическая экстраполяция кривой FLC влево и вправо
        """
        # Экстраполяция влево (между TE и PS)
        x_TE, y_TE = x_vals[0], y_vals[0]
        x_PS, y_PS = x_vals[1], y_vals[1]

        slope_left = (y_PS - y_TE) / (x_PS - x_TE) if x_PS != x_TE else 0
        intercept_left = y_TE - slope_left * x_TE

        left_limit = self.round_down_to_nearest_tenth(x_TE) - 0.1
        x_ext_left = np.linspace(left_limit, x_TE, 50)
        y_ext_left = slope_left * x_ext_left + intercept_left

        # Экстраполяция вправо (между IM и BI)
        x_IM, y_IM = x_vals[-2], y_vals[-2]
        x_BI, y_BI = x_vals[-1], y_vals[-1]

        slope_right = (y_BI - y_IM) / (x_BI - x_IM) if x_BI != x_IM else 0
        intercept_right = y_BI - slope_right * x_BI

        right_limit = self.round_up_to_nearest_tenth(x_BI) + 0.1
        x_ext_right = np.linspace(x_BI, right_limit, 50)
        y_ext_right = slope_right * x_ext_right + intercept_right

        return (x_ext_left, y_ext_left), (x_ext_right, y_ext_right)

    def plot_FLC(self, extrapolate_left=False, extrapolate_right=False,
                 title="Forming Limit Curve (FLC)", show_legend=False):
        """
        Строит диаграмму FLC с динамической экстраполяцией
        """
        points = self.get_FLC_points()
        labels = ['TE', 'PS', 'IM', 'BI']
        x_vals = [points[key][0] for key in labels]
        y_vals = [points[key][1] for key in labels]

        left_line, right_line = self.extrapolate_FLC_dynamically(x_vals, y_vals)

        plt.figure(figsize=(10, 6))

        # Основные точки FLC
        plt.plot(x_vals, y_vals, 'black', label='Forming Limit Curve')
        plt.scatter(x_vals, y_vals, color='black', zorder=5)

        # Подписи точек
        plt.plot([], [], ' ', label='TE - uniaxial tension necking point')
        plt.plot([], [], ' ', label=f"       $\\varepsilon_1$ = {round(y_vals[0], 3)}, $\\varepsilon_2$ = {round(x_vals[0], 3)}")

        plt.plot([], [], ' ', label='PS - plane strain point')
        plt.plot([], [], ' ', label=f"       $\\varepsilon_1$ = {round(y_vals[1], 3)}, $\\varepsilon_2$ = {round(x_vals[1], 3)}")

        plt.plot([], [], ' ', label='IM - intermediate biaxial stretch point')
        plt.plot([], [], ' ', label=f"       $\\varepsilon_1$ = {round(y_vals[2], 3)}, $\\varepsilon_2$ = {round(x_vals[2], 3)}")

        plt.plot([], [], ' ', label='BI - biaxial stretch point')
        plt.plot([], [], ' ', label=f"       $\\varepsilon_1$ = {round(y_vals[3], 3)}, $\\varepsilon_2$ = {round(x_vals[3], 3)}")

        # Условное построение экстраполяции
        if extrapolate_left:
            plt.plot(left_line[0], left_line[1], 'k--')

        if extrapolate_right:
            plt.plot(right_line[0], right_line[1], 'k--')

        # Аннотации точек
        for i, label in enumerate(labels):
            plt.annotate(label, (x_vals[i], y_vals[i]), textcoords="offset points", xytext=(0, 10), ha='center')

        # Настройки графика
        plt.xlabel("Minor Strain $\\varepsilon_2$", fontsize=16)
        plt.ylabel("Major Strain $\\varepsilon_1$", fontsize=16)
        #plt.title(title, fontsize=16)
        plt.grid(True)

        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))

        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)

        # Автоматический масштаб осей
        plt.xlim(
            left_line[0][0] - 0.05 if extrapolate_left else min(x_vals) - 0.05,
            right_line[0][-1] + 0.05 if extrapolate_right else max(x_vals) + 0.05
        )
        plt.ylim(0.0, max(y_vals) + 0.1)

        if show_legend:
            plt.legend(fontsize=11)

        plt.tight_layout()
        plt.show()





# ======================================
# Пример использования
# ======================================

if __name__ == "__main__":
    A80 = 40.8
    r0, r45, r90 = 1.769, 1.661, 2.225
    t = 1.2

    flc_model = FLCPredictor(A80=A80, r0=r0, r45=r45, r90 = r90, t=t)

    print("Прогнозируемые точки FLC:")
    for key, val in flc_model.get_FLC_points().items():
        print(f"{key}: ε₂={val[0]:.4f}, ε₁={val[1]:.4f}")

    flc_model.plot_FLC(extrapolate_left=False, extrapolate_right=False,
                       title=f"Forming Limit Diagram (A80={A80}%, t={t} mm)", show_legend=True)