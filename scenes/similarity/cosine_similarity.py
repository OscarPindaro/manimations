from manim import *
import numpy as np
from manimations.colors import OneDarkClassicPalette, OneDarkVividPalette

Palette = OneDarkVividPalette


class CircleWithCenter(VGroup):
    def __init__(
        self,
        radius: float | None = None,
        color: ParsableManimColor = WHITE,
        center_color: ParsableManimColor = WHITE,
        **kwargs,
    ) -> None:
        VGroup.__init__(self, **kwargs)
        self.circle = Circle(radius=radius, color=color, **kwargs)
        self.center_point = Dot(self.circle.get_center(), color=center_color)
        self.add(self.circle, self.center_point)

    def get_center(self):
        return self.circle.get_center()


class CosineSimilarityGroup(VGroup):
    def __init__(self, angle=0, radius=2, **kwargs):
        super().__init__(**kwargs)
        # Green vector (fixed)
        self.green_vec = np.array([1, 0, 0])
        # Red vector rotates
        self.red_vec = np.array([np.cos(angle), np.sin(angle), 0])
        # Circle
        self.circle = CircleWithCenter(
            radius=radius, color=WHITE, center_color=Palette.WHITE
        )
        # Vectors
        self.green_arrow = Vector(self.green_vec * radius, color=Palette.GREEN)
        self.red_arrow = Vector(self.red_vec * radius, color=Palette.RED)
        # Projections
        dot = np.dot(self.red_vec, self.green_vec)
        proj_point = dot * self.green_vec * radius
        self.proj_line = Line(
            self.circle.get_center(),
            proj_point,
            stroke_width=6,
            color=Palette.YELLOW,
            z_index=1,
        )
        self.vert_line = DashedLine(
            self.red_arrow.get_end(),
            proj_point,
            color=Palette.BLUE,
        )
        self.dot = Dot(
            proj_point,
            color=Palette.LIGHT_BLUE,
        )
        # Labels
        self.red_label = MathTex("\\vec{r}", color=Palette.RED).next_to(
            self.red_arrow.get_end(), UP
        )
        self.green_label = MathTex("\\vec{g}", color=Palette.GREEN).next_to(
            self.green_arrow.get_end(), RIGHT
        )
        # Group
        self.add(
            self.circle,
            self.red_arrow,
            self.green_arrow,
            self.proj_line,
            self.vert_line,
            self.dot,
            self.red_label,
            self.green_label,
        )
        self.radius = radius
        self.angle = angle
        self.update_group(angle)

    def scale_red_arrow(self, magnitude):
        self.red_arrow.put_start_and_end_on(
            self.circle.get_center(),
            self.circle.get_center() + self.red_vec * self.radius * magnitude,
        )
        return self.red_arrow

    def scale_green_arrow(self, magnitude):
        self.green_arrow.put_start_and_end_on(
            self.circle.get_center(),
            self.circle.get_center() + self.green_vec * self.radius * magnitude,
        )
        return self.green_arrow

    def update_vector_lenghts(self, red_lenght, green_lenght):
        self.red_arrow.put_start_and_end_on(
            self.circle.get_center(),
            self.circle.get_center() + self.red_vec * self.radius * red_lenght,
        )
        self.green_arrow.put_start_and_end_on(
            self.circle.get_center(),
            self.circle.get_center() + self.green_vec * self.radius * green_lenght,
        )

    def update_group(self, angle):
        self.angle = angle
        center = self.circle.get_center()
        self.red_vec = np.array([np.cos(angle), np.sin(angle), 0])
        red_end = center + self.red_vec * self.radius
        green_end = center + self.green_vec * self.radius
        self.red_arrow.put_start_and_end_on(center, red_end)
        self.green_arrow.put_start_and_end_on(center, green_end)
        dot = np.dot(self.red_vec, self.green_vec)
        proj_point = center + dot * self.green_vec * self.radius
        self.proj_line.put_start_and_end_on(center, proj_point)
        self.vert_line.put_start_and_end_on(red_end, proj_point)
        self.dot.move_to(proj_point)
        self.red_label.next_to(red_end, UP)
        self.green_label.next_to(green_end, RIGHT)

    def get_cosine_similarity(self):
        return np.dot(self.red_vec, self.green_vec) / (
            np.linalg.norm(self.red_vec) * np.linalg.norm(self.green_vec)
        )


class CosineSimilarityScene(Scene):
    STARTING_ANGLE = PI / 4
    RADIUS = 2
    EPS = 0.1  # if the projection becose Null, something bad happens

    def construct(self):
        self.camera.background_color = Palette.DARK_BACKGROUND
        group = CosineSimilarityGroup(angle=self.STARTING_ANGLE, radius=self.RADIUS)
        group.shift(2.5 * LEFT)
        group.update_vector_lenghts(3, 1.5)
        self.add(group)
        angle_tracker = ValueTracker(self.STARTING_ANGLE)
        red_length_tracker = ValueTracker(3)
        green_length_tracker = ValueTracker(1.5)
        # Cosine similarity formula
        formula = MathTex(
            "f(\\vec{a}, \\vec{b}) = "
            "\\frac{\\vec{a} \\cdot \\vec{b}}{|\\vec{a}||\\vec{b}|} = ",
            font_size=36,
            color=Palette.WHITE,
        ).to_edge(RIGHT)
        # Value display
        similarity_value = DecimalNumber(
            group.get_cosine_similarity(),
            num_decimal_places=2,
            font_size=36,
            color=Palette.YELLOW,
        )
        similarity_value.next_to(formula, RIGHT)
        text_group = (
            VGroup(
                formula,
                similarity_value,
            )
            .arrange(RIGHT)
            .to_edge(RIGHT)
        ).shift(0.5 * LEFT)
        self.add(text_group)

        title = MathTex("Cosine Similarity", font_size=48, color=Palette.WHITE).to_edge(
            UP
        )
        self.add(title)

        # Updater for similarity value
        def update_similarity_value(mob):
            mob.set_value(group.get_cosine_similarity())
            mob.next_to(formula, RIGHT)

        similarity_value.add_updater(update_similarity_value)
        self.add(formula, similarity_value, title)
        # Animation logic
        group.add_updater(
            lambda mob: mob.update_vector_lenghts(
                red_length_tracker.get_value(), green_length_tracker.get_value()
            )
        )
        self.play(
            red_length_tracker.animate.set_value(1),
            green_length_tracker.animate.set_value(1),
        )
        group.add_updater(lambda mob: mob.update_group(angle_tracker.get_value()))
        self.play(
            angle_tracker.animate.set_value(7 / 6 * PI),
            run_time=4,
        )
        self.wait()
        self.play(
            angle_tracker.animate.set_value(2 / 3 * PI),
            run_time=2,
        )
        self.wait()
        self.play(
            angle_tracker.animate.set_value(self.STARTING_ANGLE + TAU),
            run_time=2,
        )
        self.wait()
