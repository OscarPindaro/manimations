from manim import *
import numpy as np
from manimations.colors import OneDarkClassicPalette, OneDarkVividPalette

Palette = OneDarkVividPalette


class DotProductGroup(VGroup):
    def __init__(
        self,
        angle=PI / 4,
        vec_a_magnitude=2,
        vec_b_magnitude=2,
        show_arc=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # State vectors (vec_a is green/fixed, vec_b is red/rotatable)
        self.vec_a = np.array([1, 0, 0])  # Fixed horizontal vector
        self.vec_b = np.array([np.cos(angle), np.sin(angle), 0])  # Rotatable vector
        self.vec_a_magnitude = vec_a_magnitude
        self.vec_b_magnitude = vec_b_magnitude
        self.angle = angle
        self.show_arc = show_arc

        # Origin dot (white center point)
        self.dot = Dot(ORIGIN, color=Palette.WHITE)

        # Vectors
        self.vec_a_arrow = Vector(
            self.vec_a * self.vec_a_magnitude, color=Palette.GREEN
        )
        self.vec_b_arrow = Vector(self.vec_b * self.vec_b_magnitude, color=Palette.RED)

        # Calculate projection
        dot_product = np.dot(self.vec_b, self.vec_a)
        self.projection_point_pos = dot_product * self.vec_a * self.vec_a_magnitude

        # Projection visualization
        self.projection_point = Dot(self.projection_point_pos, color=Palette.LIGHT_BLUE)
        self.projection_line = Line(
            ORIGIN,
            self.projection_point_pos,
            stroke_width=6,
            color=Palette.YELLOW,
            z_index=1,
        )
        self.vert_line = DashedLine(
            self.vec_b_arrow.get_end(), self.projection_point_pos, color=Palette.BLUE
        )

        # Angle arc (optional)
        self.angle_arc = None
        if self.show_arc:
            arc_radius = min(self.vec_a_magnitude, self.vec_b_magnitude) * 0.3
            # Ensure arc radius is reasonable
            arc_radius = max(0.2, min(arc_radius, 1.0))
            self.angle_arc = ArcBetweenPoints(
                start=self.dot.get_center() + self.vec_a * arc_radius,
                end=self.dot.get_center() + self.vec_b * arc_radius,
                radius=arc_radius,
                color=Palette.WHITE,
                stroke_width=8,
            )
            # self.angle_arc.move_arc_center_to(self.dot.get_center())

        # Labels
        self.vec_a_label = MathTex("\\vec{a}", color=Palette.GREEN).next_to(
            self.vec_a_arrow.get_end(), DOWN
        )
        self.vec_b_label = MathTex("\\vec{b}", color=Palette.RED).next_to(
            self.vec_b_arrow.get_end(), UP
        )

        # Add all elements to the group
        elements_to_add = [
            self.vec_a_arrow,
            self.vec_b_arrow,
            self.projection_line,
            self.vert_line,
            self.dot,
            self.projection_point,
            self.vec_a_label,
            self.vec_b_label,
        ]

        if self.angle_arc is not None:
            elements_to_add.insert(0, self.angle_arc)

        self.add(*elements_to_add)

        self.update_group()

    def update_angle(self, angle):
        """Update the angle between vectors (vec_a remains fixed)"""
        self.angle = angle
        self.vec_b = np.array([np.cos(angle), np.sin(angle), 0])
        self.update_group()

    def update_vec_b_magnitude(self, magnitude):
        """Update the magnitude of vec_b (red vector)"""
        self.vec_b_magnitude = magnitude
        self.update_group()

    def update_vec_a_magnitude(self, magnitude):
        """Update the magnitude of vec_a (green vector)"""
        self.vec_a_magnitude = magnitude
        self.update_group()

    def update_group(self):
        """Update all visualization elements based on current state"""
        center = self.dot.get_center()

        # Update vector endpoints
        vec_a_end = center + self.vec_a * self.vec_a_magnitude
        vec_b_end = center + self.vec_b * self.vec_b_magnitude

        # Update arrows
        self.vec_a_arrow.put_start_and_end_on(center, vec_a_end)
        self.vec_b_arrow.put_start_and_end_on(center, vec_b_end)

        # Calculate projection
        dot_product = np.dot(
            self.vec_b * self.vec_b_magnitude, self.vec_a * self.vec_a_magnitude
        )
        # given that a*b = |a||b|cos(theta)
        # i want only the orizontal component of |b|, therefore
        # |b|cos(theta)
        self.projection_point_pos = (
            center + dot_product * self.vec_a / self.vec_a_magnitude
        )
        # self.projection_point_pos = center + dot_product * self.vec_a

        # Update projection visualization
        self.projection_line.put_start_and_end_on(center, self.projection_point_pos)
        self.vert_line.put_start_and_end_on(vec_b_end, self.projection_point_pos)
        self.projection_point.move_to(self.projection_point_pos)

        # Update angle arc
        if self.angle_arc is not None:
            arc_radius = min(self.vec_a_magnitude, self.vec_b_magnitude) * 0.3
            # Ensure arc radius is reasonable
            arc_radius = max(0.2, min(arc_radius, 1.0))

            # Create new arc with updated angle
            if (self.vec_b - self.vec_a)[1] > 0:
                new_arc = ArcBetweenPoints(
                    start=center + self.vec_a * arc_radius,
                    end=center + self.vec_b * arc_radius,
                    color=Palette.WHITE,
                    stroke_width=8,
                )
            else:
                new_arc = ArcBetweenPoints(
                    start=center + self.vec_b * arc_radius,
                    end=center + self.vec_a * arc_radius,
                    color=Palette.WHITE,
                    stroke_width=8,
                )
            self.angle_arc.become(new_arc)

        # Update labels
        self.vec_a_label.next_to(vec_a_end, DOWN)
        self.vec_b_label.next_to(vec_b_end, UP)

    def get_dot_product(self):
        """Calculate the dot product of the current vectors"""
        return np.dot(
            self.vec_b * self.vec_b_magnitude, self.vec_a * self.vec_a_magnitude
        )


class DotProductScene(Scene):
    STARTING_ANGLE = PI / 3
    VEC_A_MAGNITUDE = 1
    VEC_B_MAGNITUDE = 2.5

    def construct(self):
        self.camera.background_color = Palette.DARK_BACKGROUND

        # Create dot product group
        dot_product = DotProductGroup(
            angle=self.STARTING_ANGLE,
            vec_a_magnitude=self.VEC_A_MAGNITUDE,
            vec_b_magnitude=self.VEC_B_MAGNITUDE,
        )
        dot_product.shift(2.5 * LEFT)
        self.add(dot_product)

        # Trackers for animation
        angle_tracker = ValueTracker(self.STARTING_ANGLE)
        vec_a_magnitude_tracker = ValueTracker(self.VEC_A_MAGNITUDE)
        vec_b_magnitude_tracker = ValueTracker(self.VEC_B_MAGNITUDE)

        # Dot product formula
        formula = MathTex(
            "f(\\vec{a}, \\vec{b}) = \\vec{a} \\cdot \\vec{b} = ",
            font_size=36,
            color=Palette.WHITE,
        ).to_edge(RIGHT)

        # Value display
        dot_product_value = DecimalNumber(
            dot_product.get_dot_product(),
            num_decimal_places=2,
            font_size=36,
            color=Palette.YELLOW,
        )
        dot_product_value.next_to(formula, RIGHT)

        text_group = (
            VGroup(formula, dot_product_value)
            .arrange(RIGHT)
            .to_edge(RIGHT)
            .shift(0.5 * LEFT)
        )
        self.add(text_group)

        dot_product.shift(DOWN * 0.5)
        text_group.shift(DOWN * 0.5)
        # Title
        title = Text("Dot Product", font_size=48, color=Palette.WHITE).to_edge(UP)
        self.add(title)

        # Updater for dot product value
        def update_dot_product_value(mob):
            mob.set_value(dot_product.get_dot_product())
            mob.next_to(formula, RIGHT)

        dot_product_value.add_updater(update_dot_product_value)

        # Animation logic with updaters
        dot_product.add_updater(lambda mob: mob.update_angle(angle_tracker.get_value()))
        dot_product.add_updater(
            lambda mob: mob.update_vec_a_magnitude(vec_a_magnitude_tracker.get_value())
        )
        dot_product.add_updater(
            lambda mob: mob.update_vec_b_magnitude(vec_b_magnitude_tracker.get_value())
        )

        # Animations
        self.play(
            vec_a_magnitude_tracker.animate.set_value(1.5),
            vec_b_magnitude_tracker.animate.set_value(1.5),
            run_time=3,
        )

        self.play(angle_tracker.animate.set_value(5 * PI / 6), run_time=3)

        self.play((vec_a_magnitude_tracker.animate.set_value(2.5)))
        self.play(
            angle_tracker.animate.set_value(-PI / 6),
            vec_b_magnitude_tracker.animate.set_value(4),
            run_time=3,
        )
        self.wait(0.5)
        self.play(
            angle_tracker.animate.set_value(self.STARTING_ANGLE),
            vec_a_magnitude_tracker.animate.set_value(self.VEC_A_MAGNITUDE),
            vec_b_magnitude_tracker.animate.set_value(self.VEC_B_MAGNITUDE),
            run_time=3,
        )
        self.wait()
