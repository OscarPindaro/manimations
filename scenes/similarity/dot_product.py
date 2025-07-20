from manim import *
from manim import Vector
from manim.typing import Vector2D, Vector3D
import numpy as np
import math
from typing import List, Tuple
from manimations.colors import OneDarkClassicPalette
from dataclasses import dataclass


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'"""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


@dataclass
class VectorProjectionState:
    """Holds all the state for the vector projection animation"""

    # Base vectors
    base_vector: np.ndarray
    initial_moving_vector: np.ndarray

    # Manim objects
    base_arrow: Vector
    moving_arrow: Vector
    projection_line: Line
    vertical_line: DashedLine
    projection_dot: Dot
    base_label: MathTex
    moving_label: MathTex

    # Trackers
    angle_tracker: ValueTracker
    magnitude_tracker: ValueTracker

    def __post_init__(self):
        # Calculate initial values
        initial_angle = angle_between(self.initial_moving_vector, RIGHT)
        initial_magnitude = np.linalg.norm(self.initial_moving_vector)

        # Set tracker initial values
        self.angle_tracker.set_value(initial_angle)
        self.magnitude_tracker.set_value(initial_magnitude)
        # z indexing
        self.base_arrow.set_z_index(0)
        self.moving_arrow.set_z_index(0)
        self.projection_line.set_z_index(-1)
        self.vertical_line.set_z_index(1)
        self.projection_dot.set_z_index(2)

    def calculate_projection(
        self, vector_a: np.ndarray, vector_b: np.ndarray
    ) -> np.ndarray:
        """Calculate the projection of vector_a onto vector_b"""
        dot_product = np.dot(vector_a, vector_b)
        magnitude_b_squared = np.dot(vector_b, vector_b)
        projection_scalar = dot_product / magnitude_b_squared
        projection_point = projection_scalar * vector_b
        return projection_point

    def get_current_moving_vector(self) -> np.ndarray:
        """Get the current position of the moving vector based on trackers"""
        angle = self.angle_tracker.get_value()
        magnitude = self.magnitude_tracker.get_value()
        return np.array([magnitude * np.cos(angle), magnitude * np.sin(angle), 0])

    def update_all_objects(
        self,
    ) -> Tuple[Vector, Line, DashedLine, Dot, MathTex, MathTex]:
        """Update all objects based on current tracker values. Returns updated objects."""
        # Get current moving vector position
        current_moving_vector = self.get_current_moving_vector()

        # Calculate projection
        projection_point = self.calculate_projection(
            current_moving_vector, self.base_vector
        )

        # Create updated objects (don't modify originals, create new ones)
        updated_moving_arrow = Vector(
            current_moving_vector, color=OneDarkClassicPalette.RED
        )

        updated_projection_line = Line(
            start=ORIGIN, end=projection_point, color=OneDarkClassicPalette.YELLOW
        )

        updated_vertical_line = DashedLine(
            start=current_moving_vector,
            end=projection_point,
            color=OneDarkClassicPalette.BLUE,
        )

        updated_projection_dot = Dot(
            projection_point, color=OneDarkClassicPalette.LIGHT_BLUE
        )

        updated_base_label = MathTex(
            "\\vec{a}", color=OneDarkClassicPalette.GREEN
        ).next_to(self.base_arrow.get_end(), DOWN)

        updated_moving_label = MathTex(
            "\\vec{b}", color=OneDarkClassicPalette.RED
        ).next_to(updated_moving_arrow.get_end(), UP)

        return (
            updated_moving_arrow,
            updated_projection_line,
            updated_vertical_line,
            updated_projection_dot,
            updated_base_label,
            updated_moving_label,
        )


class VectorScene(Scene):
    def construct(self):
        # Initial vector values
        base_vec = RIGHT
        moving_vec = np.array([2, 2, 0])

        # Calculate initial projection
        initial_projection = self.calculate_projection(moving_vec, base_vec)

        # Create state object
        self.state = VectorProjectionState(
            base_vector=base_vec,
            initial_moving_vector=moving_vec,
            base_arrow=Vector(base_vec, color=OneDarkClassicPalette.GREEN),
            moving_arrow=Vector(moving_vec, color=OneDarkClassicPalette.RED),
            projection_line=Line(
                start=ORIGIN, end=initial_projection, color=OneDarkClassicPalette.YELLOW
            ),
            vertical_line=DashedLine(
                start=moving_vec,
                end=initial_projection,
                color=OneDarkClassicPalette.BLUE,
            ),
            projection_dot=Dot(
                initial_projection, color=OneDarkClassicPalette.LIGHT_BLUE
            ),
            base_label=MathTex("\\vec{a}", color=OneDarkClassicPalette.GREEN),
            moving_label=MathTex("\\vec{b}", color=OneDarkClassicPalette.RED),
            angle_tracker=ValueTracker(0),
            magnitude_tracker=ValueTracker(0),
        )

        # Position labels
        self.state.base_label.next_to(self.state.base_arrow.get_end(), DOWN)
        self.state.moving_label.next_to(self.state.moving_arrow.get_end(), UP)

        # Add dot product formula
        dot_product_text = MathTex(
            "f(a,b) = \\vec{a} \\cdot \\vec{b}",
            font_size=36,
            color=OneDarkClassicPalette.WHITE,
        ).to_edge(UL)

        # Add all objects to scene
        self.add(
            self.state.base_arrow,
            self.state.moving_arrow,
            self.state.base_label,
            self.state.moving_label,
            self.state.projection_line,
            self.state.vertical_line,
            self.state.projection_dot,
            dot_product_text,
        )
        self.wait()

        # Add updaters
        self.state.moving_arrow.add_updater(
            lambda obj: obj.become(self.state.update_all_objects()[0])
        )
        self.state.projection_line.add_updater(
            lambda obj: obj.become(self.state.update_all_objects()[1])
        )
        self.state.vertical_line.add_updater(
            lambda obj: obj.become(self.state.update_all_objects()[2])
        )
        self.state.projection_dot.add_updater(
            lambda obj: obj.become(self.state.update_all_objects()[3])
        )
        self.state.base_label.add_updater(
            lambda obj: obj.become(self.state.update_all_objects()[4])
        )
        self.state.moving_label.add_updater(
            lambda obj: obj.become(self.state.update_all_objects()[5])
        )

        # Animate
        self.play(
            self.state.angle_tracker.animate.set_value(np.pi),
            self.state.magnitude_tracker.animate.set_value(1.5),
            run_time=3,
        )

        self.wait()

        # Additional animation
        self.play(
            self.state.angle_tracker.animate.set_value(2 * np.pi),
            self.state.magnitude_tracker.animate.set_value(3),
            run_time=4,
        )

        self.wait()

    def calculate_projection(
        self, vector_a: np.ndarray, vector_b: np.ndarray
    ) -> np.ndarray:
        """Calculate the projection of vector_a onto vector_b"""
        dot_product = np.dot(vector_a, vector_b)
        magnitude_b_squared = np.dot(vector_b, vector_b)
        projection_scalar = dot_product / magnitude_b_squared
        projection_point = projection_scalar * vector_b
        return projection_point
