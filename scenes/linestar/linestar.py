from manim import *
from dataclasses import dataclass, field
import numpy as np
from typing import List, Tuple, Literal

from manimations.colors import ONEDARK_CLASS_RAINBOW, COLOR_LISTS, COLORS


@dataclass
class LineStar:
    radius: float
    n: int
    colors: List[ManimColor] = field(default_factory=list)
    grouped_lines: VGroup = field(
        default_factory=lambda: VGroup(), repr=False, init=False
    )  # Hidden from repr
    lines: List[Line] = field(default_factory=list, repr=False, init=False)
    gradient: List[ManimColor] = field(default_factory=list, repr=False, init=False)

    def __post_init__(self):
        # Private numpy array that is linearly spaced from 0 to radius with n steps
        self._linear_space: np.ndarray = np.linspace(0, self.radius, self.n)

        if len(self.colors) > 0:
            # the number of colors is n*4 because in the for loop i create 4 lines
            self.gradient = color_gradient(self.colors, self.n * 4)

        # let's create the lines. I'll create them from the inner to the outer with a pattern that is hard to explai
        for i in range(len(self._linear_space)):
            inner = self._linear_space[i]
            outer = self._linear_space[len(self._linear_space) - i - 1]
            self.lines.append(
                self._create_line(
                    (0, outer, 0),
                    (inner, 0, 0),
                    color=None if len(self.gradient) == 0 else self.gradient[4 * i + 0],
                )
            )
            self.lines.append(
                self._create_line(
                    (inner, 0, 0),
                    (0, -outer, 0),
                    color=None if len(self.gradient) == 0 else self.gradient[4 * i + 1],
                )
            )
            self.lines.append(
                self._create_line(
                    (0, -outer, 0),
                    (-inner, 0, 0),
                    color=None if len(self.gradient) == 0 else self.gradient[4 * i + 2],
                )
            )
            self.lines.append(
                self._create_line(
                    (-inner, 0, 0),
                    (0, outer, 0),
                    color=None if len(self.gradient) == 0 else self.gradient[4 * i + 3],
                )
            )
        # add it to the vgroup if you need to move it
        self.grouped_lines.add(*self.lines)

    def _create_line(
        self,
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
        color: ManimColor | None = None,
    ) -> Line:
        if color is None:
            return Line(start, end)
        else:
            return Line(start, end, color=color)

    def create_lines(self) -> List[Create]:
        return [Create(line) for line in self.lines]

    def uncreate_lines(self):
        return self.animate_lines(Uncreate)

    def animate_lines(self, animation: Animation) -> List[Animation]:
        return [animation(line) for line in self.lines]

    def get_linear_space(self):
        # Public method to access the private array if needed
        return self._linear_space


class LinestarScene(Scene):
    curr_palette: List[str | ManimColor] = []
    reverse_palette: bool = False
    background_color: ManimColor | str = field(default_factory=WHITE)
    create_lag_ratio: float = 0.01

    def setup(self):
        if not self.curr_palette:
            self.curr_palette = ONEDARK_CLASS_RAINBOW
        if self.reverse_palette:
            self.curr_palette.reverse()
        self.linestar = LineStar(6, 25, colors=self.curr_palette)

    def construct(self):
        self.camera.background_color = self.background_color
        self.wait(0.25)
        self.play(
            LaggedStart(*self.linestar.create_lines(), lag_ratio=self.create_lag_ratio)
        )
        # self.play(FadeIn(Line((0,0,0), (3,3,0))))
        self.wait(0.25)
        self.play(
            LaggedStart(
                *self.linestar.uncreate_lines(), lag_ratio=self.create_lag_ratio
            )
        )


def render_linestar(
    width: int,
    height: int,
    frame_rate: float,
    output: str,
    palette: str,
    background_color: str,
    video_dir: str,
    format: Literal[None, "png", "gif", "mp4", "mov", "webm"],
):

    format = "mp4" if format is None else format
    # Generate the default output file name if none is provided
    if not output:
        # Normalize the palette and background color for the filename
        normalized_palette = palette.lower().replace(" ", "_")
        if isinstance(background_color, str):
            normalized_background = (
                background_color.lstrip("#").lower().replace(" ", "_")
            )  # If hex, strip the '#'
        elif isinstance(background_color, ManimColor):
            normalized_background = (
                background_color.to_hex().lstrip("#").lower().replace(" ", "_")
            )
        else:
            raise ValueError(
                f"Not handling the background_color type: {type(background_color)}"
            )
        scene_class_name = LinestarScene.__name__
        output = f"{scene_class_name}_{normalized_palette}_{normalized_background}"
    # Proceed with rendering the scene
    external_config = {
        "preview": False,
        "pixel_width": width,
        "pixel_height": height,
        "output_file": output,
        "frame_rate": frame_rate,
        "format": format,
    }

    if video_dir is not None:
        external_config["video_dir"] = video_dir
    with tempconfig(external_config):
        # print(config.keys())  # Debug print

        scene = LinestarScene()
        scene.curr_palette = COLOR_LISTS[palette]  # Use the validated palette
        scene.background_color = background_color  # Set the validated background color
        scene.render()
