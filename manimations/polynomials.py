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
