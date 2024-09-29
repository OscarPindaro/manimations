# %%
import sys
from pathlib import Path
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# %%
from dataclasses import dataclass
import numpy as np
from typing import Union, List, Tuple
import matplotlib.pyplot as plt

# %%
from manim import *

from manimations.colors import OneDarkClassicPalette, GoldenSunsetPalette
from manimations.polynomials import Polynomial


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
        self.max_degree = 3

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

        plane_vgroup = VGroup(self.number_plane)
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
