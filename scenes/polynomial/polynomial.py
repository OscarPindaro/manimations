# %%
import sys
from pathlib import Path

# %%
from dataclasses import dataclass
import numpy as np
from typing import Union, List, Tuple
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Polynomial:
    coefficients: tuple

    def __post_init__(self):
        object.__setattr__(
            self, "coefficients", tuple(float(c) for c in self.coefficients)
        )

    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return sum(coef * x**power for power, coef in enumerate(self.coefficients))

    def derivative(self) -> "Polynomial":
        new_coeffs = [i * coef for i, coef in enumerate(self.coefficients)][1:]
        return Polynomial(new_coeffs)

    @classmethod
    def random(
        cls, max_degree: int, min_value: float, max_value: float
    ) -> "Polynomial":
        assert max_value > min_value
        assert max_degree >= 0
        coeffs = np.random.rand(max_degree + 1) * (max_value - min_value) + min_value
        return cls(coeffs)

    def __add__(self, other: "Polynomial") -> "Polynomial":
        max_degree = max(len(self.coefficients), len(other.coefficients))
        new_coeffs = [0] * max_degree
        for i, coef in enumerate(self.coefficients):
            new_coeffs[i] += coef
        for i, coef in enumerate(other.coefficients):
            new_coeffs[i] += coef
        return Polynomial(new_coeffs)

    def __sub__(self, other: "Polynomial") -> "Polynomial":
        max_degree = max(len(self.coefficients), len(other.coefficients))
        new_coeffs = [0] * max_degree
        for i, coef in enumerate(self.coefficients):
            new_coeffs[i] += coef
        for i, coef in enumerate(other.coefficients):
            new_coeffs[i] -= coef
        return Polynomial(new_coeffs)

    def __mul__(self, scalar: float) -> "Polynomial":
        new_coeffs = [coef * scalar for coef in self.coefficients]
        return Polynomial(new_coeffs)

    def __rmul__(self, scalar: float) -> "Polynomial":
        return self.__mul__(scalar)

    def compute_error(self, x: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute the Mean Squared Error between the polynomial predictions and true values.
        """
        y_pred = self(x)
        return np.mean((y_pred - y_true) ** 2)

    def compute_gradient(self, x: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the Mean Squared Error with respect to the coefficients.
        """
        y_pred = self(x)
        error = y_pred - y_true
        gradient = np.zeros(len(self.coefficients))
        for i in range(len(self.coefficients)):
            gradient[i] = 2 * np.mean(error * x**i)
        return gradient

    def gradient_ascent_step(
        self,
        x: np.ndarray,
        y_true: np.ndarray,
        alpha: float,
        clip_val: float | None = None,
    ) -> "Polynomial":
        """
        Perform a single step of gradient ascent and return an updated Polynomial.
        """
        gradient = self.compute_gradient(x, y_true)
        if clip_val:
            gradient = np.clip(gradient, -clip_val, clip_val)
        new_coeffs = [
            coef - alpha * grad for coef, grad in zip(self.coefficients, gradient)
        ]
        return Polynomial(new_coeffs)

    def gradient_descent_step(
        self, x: np.ndarray, y_true: np.ndarray, alpha: float
    ) -> "Polynomial":
        """
        Perform a single step of gradient ascent and return an updated Polynomial.
        """
        gradient = self.compute_gradient(x, y_true)
        new_coeffs = [
            coef + alpha * grad for coef, grad in zip(self.coefficients, gradient)
        ]
        return Polynomial(new_coeffs)

    def plot(
        self, xlim: Tuple[float, float] = (-5, 5), ylim: Tuple[float, float] = None
    ) -> None:
        x = np.linspace(xlim[0], xlim[1], 1000)
        y = self(x)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, y, label=f"Polynomial: {self}")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Polynomial Plot")
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

        # Add x and y axes if they're within the plot range
        ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
        ax.axvline(x=0, color="k", linestyle="-", linewidth=0.5)

        # Adjust axis labels position for better visibility
        ax.spines["left"].set_position("zero")
        ax.spines["bottom"].set_position("zero")
        ax.spines["right"].set_color("none")
        ax.spines["top"].set_color("none")

        # Move the x-axis label to the right
        ax.xaxis.set_label_coords(1.05, 0.5)
        # Move the y-axis label to the top
        ax.yaxis.set_label_coords(0.5, 1.05)

        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend(loc="upper left")

        plt.tight_layout()
        plt.show()

    @classmethod
    def plot_polynomials(
        cls,
        polynomials: List["Polynomial"],
        X: List | None = None,
        y_true: List | None = None,
        xlim: Tuple[float, float] = (-5, 5),
        ylim: Tuple[float, float] = None,
        title: str = "Multiple Polynomial Plot",
    ):
        x = np.linspace(xlim[0], xlim[1], 1000)

        fig, ax = plt.subplots(figsize=(12, 8))
        max_degrees = max([len(p.coefficients) for p in polynomials])
        for i, poly in enumerate(polynomials):
            y = poly(x)
            label_to_write = (
                f"Polynomial {i+1}: {poly}" if max_degrees < 5 else f"Polynomial {i+1}"
            )
            if i == 0:
                ax.plot(x, y, label=label_to_write, linewidth=5, color="red")
            elif i == len(polynomials) - 1:
                ax.plot(x, y, label=label_to_write, linewidth=5, color="green")
            else:
                ax.plot(x, y)

        # plot the points
        assert (X is None and y_true is None) or (
            X is not None and y_true is not None
        ), f"X = {X}\ny_true = {y_true}"
        if X is not None:
            plt.plot(X, y_true, color="blue", marker="o", markersize=12)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

        ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
        ax.axvline(x=0, color="k", linestyle="-", linewidth=0.5)

        ax.spines["left"].set_position("zero")
        ax.spines["bottom"].set_position("zero")
        ax.spines["right"].set_color("none")
        ax.spines["top"].set_color("none")

        ax.xaxis.set_label_coords(1.05, 0.5)
        ax.yaxis.set_label_coords(0.5, 1.05)

        ax.grid(True, linestyle="--", alpha=0.7)
        # if len(polynomials) <= 3:

        ax.legend(loc="upper left")

        plt.tight_layout()
        plt.show()

    @classmethod
    def fit_closed_form(cls, x: np.ndarray, y: np.ndarray, degree: int) -> "Polynomial":
        """
        Compute the closed-form solution for linear regression.

        Args:
        x (np.ndarray): Input features (1D array)
        y (np.ndarray): Target values (1D array)
        degree (int): Degree of the polynomial to fit

        Returns:
        Polynomial: The best-fit polynomial
        """
        # Create the design matrix X
        X = np.column_stack([x**i for i in range(degree + 1)])

        # Compute the closed-form solution
        coeffs = np.linalg.inv(X.T @ X) @ X.T @ y

        return cls(coeffs)

    def __repr__(self) -> str:
        terms = []
        signs = []
        for power, coef in enumerate(self.coefficients):
            if coef != 0:
                if power == 0:
                    terms.append(f"{abs(coef):.2f}")
                elif power == 1:
                    terms.append(f"{abs(coef):.2f}x")
                else:
                    terms.append(f"{abs(coef):.2f}x^{power}")
            if power == len(self.coefficients) - 1:
                signs.append("" if coef >= 0 else "-")
            else:
                signs.append("+" if coef >= 0 else "-")
        final_str = ""
        for idx, (sign, term) in enumerate(zip(signs[::-1], terms[::-1])):
            if idx == 0:
                final_str += term if sign == "" else f"{sign} {term}"
            else:
                final_str += f" {sign} {term}"
        return final_str if final_str != "" else "0"

    def translate(self, x_shift: float = 0, y_shift: float = 0) -> "Polynomial":
        """
        Create a new polynomial that is a translation of the current one.

        For horizontal translation: p_translated(x) = p(x + x_shift)
        For vertical translation: p_translated(x) = p(x) + y_shift

        Args:
        x_shift (float): Amount to shift in the x direction (positive is left, negative is right)
        y_shift (float): Amount to shift in the y direction (positive is up, negative is down)

        Returns:
        Polynomial: A new polynomial representing the translated curve
        """
        # Start with the coefficients of the current polynomial
        new_coeffs = list(self.coefficients)

        # Horizontal translation
        if x_shift != 0:
            degree = len(self.coefficients) - 1
            shifted_coeffs = [0] * (degree + 1)
            for i, coef in enumerate(new_coeffs):
                for j in range(i, degree + 1):
                    shifted_coeffs[j] += (
                        coef * (x_shift) ** (j - i) * np.math.comb(j, i)
                    )
            new_coeffs = shifted_coeffs

        # Vertical translation
        if y_shift != 0:
            new_coeffs[0] += y_shift

        return Polynomial(new_coeffs)

    def find_x_for_y(
        self,
        y_value: float,
        x_range: Tuple[float, float] = (-10, 10),
        tol: float = 1e-6,
        max_iter: int = 1000,
    ) -> float:
        """
        Find the x value such that the polynomial evaluates to y_value, using the bisection method.

        Args:
        y_value (float): The target y value for which we want to find the corresponding x.
        x_range (Tuple[float, float]): The range within which to search for the x value.
        tol (float): The tolerance for stopping the bisection method.
        max_iter (int): The maximum number of iterations allowed.

        Returns:
        float: The x value that corresponds to the given y_value.
        """
        a, b = x_range
        fa = self(a) - y_value
        fb = self(b) - y_value

        if fa * fb > 0:
            raise ValueError(
                "The function must have different signs at the endpoints of the interval (i.e., there must be a root in the interval)."
            )

        for _ in range(max_iter):
            c = (a + b) / 2.0
            fc = self(c) - y_value

            if abs(fc) < tol:
                return c
            elif fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc

        raise ValueError(
            "Maximum iterations reached without finding a root within the desired tolerance."
        )


# %%
from manim import *

from src.colors import OneDarkClassicPalette, GoldenSunsetPalette


class PolynomialTransformation(Scene):

    def setup(self):
        self.start_poly = Polynomial(
            (
                -2.506546491237059,
                -1.5729344969576333,
                4.388175761334803,
                4.472889300820247,
                1.0947492804774148,
            )
        )
        self.x_start_poly = [
            -3,
            self.start_poly.find_x_for_y(y_value=10, x_range=[1, 2]),
        ]
        self.start_color = RED

        self.end_poly = Polynomial(
            (
                -1.5729344969576333,
                4.388175761334803,
                4.472889300820247,
                1.0947492804774148,
            )
        )
        self.x_end_poly = [
            self.end_poly.find_x_for_y(y_value=-3, x_range=[-4, -2.5]),
            self.end_poly.find_x_for_y(y_value=10, x_range=[0, 4]),
        ]
        self.end_color = GREEN

        self.x_range = [-4, 4]
        self.y_range = [-3, 10]

        self.number_plane = NumberPlane(
            x_range=self.x_range,
            y_range=self.y_range,
            y_length=6,
            x_length=12,
        )

        self.number_plane.add_coordinates()

    def construct(self):

        starting_function = self.number_plane.plot(
            lambda x: self.start_poly(x),
            x_range=self.x_start_poly,
            color=self.start_color,
        )
        print(starting_function)
        start_poly_formula = MathTex(
            "p(x) = " + str(self.start_poly), color=self.start_color
        )

        start_poly_formula.next_to(self.number_plane, UP)

        self.play(DrawBorderThenFill(self.number_plane))
        self.play(Create(starting_function), Write(start_poly_formula))

        target_function = self.number_plane.plot(
            lambda x: self.end_poly(x), x_range=self.x_end_poly, color=self.end_color
        )
        end_poly_formula = MathTex(
            "p(x) = " + str(self.end_poly), color=self.end_color
        ).move_to(start_poly_formula)

        self.play(
            Transform(starting_function, target_function),
            Transform(start_poly_formula, end_poly_formula),
        )
        self.pause(1)


class PolynomialFitting(Scene):

    def setup(self):

        # true distribution
        self.true_distribution = Polynomial(
            (
                -2.506546491237059,
                -1.5729344969576333,
                4.388175761334803,
                4.472889300820247,
                1.0947492804774148,
            )
        )
        self.x_range_true = [
            -3,
            self.true_distribution.find_x_for_y(y_value=10, x_range=[1, 2]),
        ]
        self.true_color = RED
        self.n_samples = 50
        self.X = np.random.uniform(*self.x_range_true, size=self.n_samples)
        self.y_true = self.true_distribution(self.X)

        # fitting process numbers
        self.n_steps = 10000
        self.lr0 = 0.01
        self.clip_val0 = 10
        self.max_degree = 10

        # generation
        self.save_every = 100

        # number plane
        self.x_range = [-4, 4]
        self.y_range = [-3, 10]

        self.number_plane = NumberPlane(
            x_range=self.x_range,
            y_range=self.y_range,
            y_length=6,
            x_length=12,
        )

        self.number_plane.add_coordinates()

    def construct(self):

        polys, errors = self.gradient_ascent_fit()

        # first let's add the true function with the datapoints
        true_function = self.number_plane.plot(
            lambda x: self.true_distribution(x),
            x_range=self.x_range_true,
            color=self.true_color,
        )

        sampled_points_anim = [
            Create(
                Dot(
                    self.number_plane.coords_to_point(p, self.true_distribution(p), 0),
                    color=GOLD,
                )
            )
            for p in self.X
        ]

        self.play(
            DrawBorderThenFill(self.number_plane),
            Create(true_function),
            sampled_points_anim,
        )

        animations = [
            Create(
                self.number_plane.plot(
                    lambda x: estimator(x), x_range=self.x_range_true
                )
            )
            for estimator in polys
        ]

        self.play(LaggedStart(animations, lag_ratio=0.2), run_time=5)

    def gradient_ascent_fit(self) -> Tuple[np.ndarray, np.ndarray]:
        import math

        errors: List[float] = []
        fitted_polys = []

        clip_val = self.clip_val0

        estimator = Polynomial.random(self.max_degree, -5, 5)
        # estimator = Polynomial((1,)*5)

        # first iteration
        fitted_polys.append(estimator)
        errors.append(estimator.compute_error(self.X, self.y_true))

        for i in range(self.n_steps):
            lr = self.lr0 / math.log(i + 2)
            # lr = self.lr0
            estimator = estimator.gradient_ascent_step(
                self.X, self.y_true, lr, clip_val
            )
            fitted_polys.append(estimator)
            errors.append(estimator.compute_error(self.X, self.y_true))

        return fitted_polys[:: self.save_every] + [fitted_polys[-1]], errors[
            :: self.save_every
        ] + [errors[-1]]


class StraightLinePoly(Scene):

    def setup(self):
        self.start_poly = Polynomial((-1, 2))
        self.x_start_poly = [-4, 4]
        self.start_color = OneDarkClassicPalette.RED

        self.end_poly = Polynomial((5, -3))
        self.x_end_poly = [-3, 10]
        self.end_color = OneDarkClassicPalette.GREEN

        self.x_range = [-4, 4]
        self.y_range = [-3, 10]

        self.number_plane = NumberPlane(
            x_range=self.x_range,
            y_range=self.y_range,
            y_length=6,
            x_length=12,
            background_line_style={
                "stroke_color": OneDarkClassicPalette.LIGHT_BLUE,
                "stroke_width": 4,
                "stroke_opacity": 0.6,
            },
        )

        self.number_plane.add_coordinates()

    def construct(self):

        # self.camera.background_color = OneDarkClassicPalette.DARK_GRAY
        self.camera.background_color = OneDarkClassicPalette.DARK_BACKGROUND

        starting_function = self.number_plane.plot(
            lambda x: self.start_poly(x),
            x_range=self.x_start_poly,
            color=self.start_color,
        )
        print(starting_function)
        start_poly_formula = MathTex(
            "p(x) = " + str(self.start_poly), color=self.start_color
        )

        start_poly_formula.next_to(self.number_plane, UP)

        self.play(DrawBorderThenFill(self.number_plane))
        self.play(Create(starting_function), Write(start_poly_formula))

        target_function = self.number_plane.plot(
            lambda x: self.end_poly(x), x_range=self.x_end_poly, color=self.end_color
        )
        end_poly_formula = MathTex(
            "p(x) = " + str(self.end_poly), color=self.end_color
        ).move_to(start_poly_formula)

        self.play(
            Transform(starting_function, target_function),
            Transform(start_poly_formula, end_poly_formula),
        )
        self.pause(1)

        all_objects = VGroup(
            starting_function,
            start_poly_formula,
            target_function,
            end_poly_formula,
            self.number_plane,
        )
        self.play(Uncreate(all_objects), run_time=2)


class ParabolePoly(Scene):

    def setup(self):
        self.start_poly = Polynomial((-1, 2, 3))
        self.x_start_poly = [-4, 4]
        self.start_color = OneDarkClassicPalette.RED

        self.end_poly = Polynomial((5, -3, -2))
        self.x_end_poly = [-3, 10]
        self.end_color = OneDarkClassicPalette.GREEN

        self.x_range = [-4, 4]
        self.y_range = [-3, 10]

        self.number_plane = NumberPlane(
            x_range=self.x_range,
            y_range=self.y_range,
            y_length=6,
            x_length=12,
            background_line_style={
                "stroke_color": OneDarkClassicPalette.LIGHT_BLUE,
                "stroke_width": 4,
                "stroke_opacity": 0.6,
            },
        )

        self.number_plane.add_coordinates()

    def construct(self):

        # self.camera.background_color = OneDarkClassicPalette.DARK_GRAY
        self.camera.background_color = OneDarkClassicPalette.DARK_BACKGROUND

        starting_function = self.number_plane.plot(
            lambda x: self.start_poly(x),
            x_range=self.x_start_poly,
            color=self.start_color,
        )
        print(starting_function)
        start_poly_formula = MathTex(
            "p(x) = " + str(self.start_poly), color=self.start_color
        )

        start_poly_formula.next_to(self.number_plane, UP)

        self.play(DrawBorderThenFill(self.number_plane))
        self.play(Create(starting_function), Write(start_poly_formula))

        target_function = self.number_plane.plot(
            lambda x: self.end_poly(x), x_range=self.x_end_poly, color=self.end_color
        )
        end_poly_formula = MathTex(
            "p(x) = " + str(self.end_poly), color=self.end_color
        ).move_to(start_poly_formula)

        self.play(
            Transform(starting_function, target_function),
            Transform(start_poly_formula, end_poly_formula),
        )
        self.pause(1)

        all_objects = VGroup(
            starting_function,
            start_poly_formula,
            target_function,
            end_poly_formula,
            self.number_plane,
        )
        self.play(Uncreate(all_objects), run_time=2)


from typing import Literal


class RandomCubics(Scene):

    def setup(self):
        # self.start_poly = Polynomial((-1, 2,3,4))
        # self.x_start_poly=[-4, 4]
        # self.start_color = GoldenSunsetPalette.DARK_RED

        # self.end_poly =  Polynomial((5,-3, -2))
        # self.x_end_poly = [-3,10]
        # self.end_color = GoldenSunsetPalette.DARK_YELLOW
        self.n_polys = 5

        self.palette: Literal["dark", "light"] = "light"

        self.x_range = [-4, 4]
        self.y_range = [-3, 10]

        self.number_plane = NumberPlane(
            x_range=self.x_range,
            y_range=self.y_range,
            y_length=6,
            x_length=12,
            background_line_style={
                "stroke_color": GoldenSunsetPalette.YELLOW_B,
                "stroke_width": 4,
                "stroke_opacity": 0.6,
            },
        )

        self.number_plane.add_coordinates()
        if self.palette == "light":
            self.number_plane.get_axes().set_color("#000000")

    def construct(self):

        # self.camera.background_color = OneDarkClassicPalette.DARK_GRAY
        self.camera.background_color = (
            GoldenSunsetPalette.LIGHT_BACKGROUND
            if self.palette == "light"
            else GoldenSunsetPalette.DARK_BACKGROUND
        )

        plane_vgroup = VGroup(self.number_plane, self.axis_labels)
        self.play(DrawBorderThenFill(plane_vgroup))

        function_colors = [
            GoldenSunsetPalette.LIGHT_RED,
            GoldenSunsetPalette.DARK_RED,
            GoldenSunsetPalette.ORANGE,
        ]
        old_function = None
        old_text = None
        for i in range(self.n_polys):
            poly = Polynomial.random(3, -1, 1)
            poly_color = function_colors[i % len(function_colors)]
            poly_function = self.number_plane.plot(
                lambda x: poly(x), x_range=self.x_range, color=poly_color
            )
            poly_text = MathTex("p(x) = " + str(poly), color=poly_color).next_to(
                self.number_plane, UP
            )

            if old_function is None:
                self.play([Create(poly_function), Write(poly_text)])
            else:
                self.remove(old_function)
                self.play(
                    ReplacementTransform(old_function, poly_function),
                    ReplacementTransform(old_text, poly_text),
                    run_time=1.5,
                )
                self.pause(0.5)

            old_function = poly_function
            old_text = poly_text

        self.pause(1)

        all_objects = VGroup(poly_function, poly_text, self.number_plane)
        self.play(Uncreate(all_objects), run_time=2)


class PolyGraphScene(Scene):

    def setup(self):
        self.poly = Polynomial(
            (
                -2.506546491237059,
                -1.5729344969576333,
                4.388175761334803,
                4.472889300820247,
                1.0947492804774148,
            )
        )

        self.x_range = [-4, 4]

    def construct(self):
        axes = Axes(
            x_range=self.x_range,
            y_range=[-3, 10],
            # x_length=14,
            # y_length=8,
            tips=False,
            # axis_config={
            #     "include_tip":True,
            #     "numbers_to_exclude":[0]
            # }
        ).add_coordinates()

        # positioning
        axes = axes.center()

        # now labels
        axis_labels = axes.get_axis_labels(x_label="x", y_label="f(x)")

        # apply function
        graph = axes.plot(lambda x: self.poly(x), x_range=self.x_range, color=YELLOW)

        grouped_graph = VGroup(axes, axis_labels, graph)

        self.play(DrawBorderThenFill(axes), Write(axis_labels), run_time=1)
        self.play(Create(graph), run_time=1)
        self.play(grouped_graph.animate.shift(UL))
        self.play(
            grouped_graph.animate.rotate(
                20,
            )
        )


class PolyNumberPlaneScene(Scene):

    def setup(self):
        self.poly = Polynomial(
            (
                -2.506546491237059,
                -1.5729344969576333,
                4.388175761334803,
                4.472889300820247,
                1.0947492804774148,
            )
        )

        self.x_range = [-4, 3]

    def construct(self):
        my_plane = NumberPlane(
            x_range=self.x_range,
            y_range=[-3, 10],
            y_length=6,
            x_length=12,
        )

        my_plane.add_coordinates()

        my_function = my_plane.plot(
            lambda x: self.poly(x),
            x_range=self.x_range,
            color=color_gradient([GREEN_B], 10),
        )

        function_expression = MathTex(str(self.poly))

        function_expression.next_to(my_plane, UP)

        area = my_plane.get_area(my_function, color=[RED, WHITE])

        horiz_line = Line(
            start=my_plane.c2p(0, my_function.underlying_function(-3)),
            end=my_plane.c2p(-3, my_function.underlying_function(-3)),
            color=[BLUE, YELLOW],
        )
        horiz_line.set_color_by_gradient([BLUE, YELLOW])

        my_function.set_color_by_gradient([GREEN_B, RED])
        print(type(area))

        self.play(DrawBorderThenFill(my_plane))
        self.add(area)
        self.play(Create(my_function), Write(function_expression), Create(horiz_line))
        # self.play(FadeIn(area))
