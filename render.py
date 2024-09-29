import click
from manim import *
from scenes.linestar.linestar import LineStar, LinestarScene, render_linestar
from src.colors import COLOR_LISTS, COLORS, MANIM_COLORS
import re
import pathlib
from typing import Literal, Tuple


# Create a custom Click type for hex color or named color validation
class ColorType(click.ParamType):
    name = "color"

    def convert(self, value, param, ctx):
        # Check if the value is a valid hex color
        if re.match(r"^#([A-Fa-f0-9]{6})$", value):
            return value
        # Check if the value exists in all_colors or all_manim_colors
        elif value in COLORS:
            return COLORS[value]
        elif value in MANIM_COLORS:
            return MANIM_COLORS[value]
        else:
            self.fail(
                f"'{value}' is not a valid hex color or known color. Expected a hex format like '#FFFFFF' or a known color name from COLORS or MANIM_COLORS."
            )


# Create a custom palette validator callback
def validate_palette(ctx, param, value):
    if value in COLOR_LISTS:
        return value
    else:
        raise click.BadParameter(
            f"'{value}' is not a valid palette. Choose from: {list(COLOR_LISTS.keys())}"
        )


# Create the CLI group
@click.group()
def cli():
    pass


# Add a 'render' command to the CLI
@cli.command("linestar")
@click.option("--width", "-w", default=720, type=int, help="Width of the output video.")
@click.option(
    "--height", "-h", default=720, type=int, help="Height of the output video."
)
@click.option(
    "--frame-rate", "-f", default=60, type=int, help="Frame rate of the output video."
)
@click.option("--output", "-o", type=str, help="Output file name.")
@click.option(
    "--palette",
    "-p",
    required=True,
    callback=validate_palette,
    help="Color palette to use.",
)
@click.option(
    "--background-color",
    "-b",
    required=True,
    type=ColorType(),
    help="Background color (hex or named color).",
)
@click.option(
    "--video-dir",
    "-v",
    type=click.Path(dir_okay=True, file_okay=False),
    help="Directory where the video is saved",
)
@click.option(
    "--format",
    default=None,
    type=click.Choice([None, "png", "gif", "mp4", "mov", "webm"]),
    help="Export format for the animation",
)
def linestar(
    width: int,
    height: int,
    frame_rate: float,
    output: str,
    palette: str,
    background_color: str,
    video_dir: str,
    format: Literal[None, "png", "gif", "mp4", "mov", "webm"],
):
    render_linestar(
        width=width,
        height=height,
        frame_rate=frame_rate,
        output=output,
        palette=palette,
        background_color=background_color,
        video_dir=video_dir,
        format=format,
    )


# Add a 'render-multiple' command to the CLI
@cli.command("render-multiple")
@click.option("--width", "-w", default=720, type=int, help="Width of the output video.")
@click.option(
    "--height", "-h", default=720, type=int, help="Height of the output video."
)
@click.option(
    "--frame-rate", "-f", default=60, type=int, help="Frame rate of the output video."
)
@click.option(
    "--video-dir",
    "-v",
    type=click.Path(file_okay=False, dir_okay=True),
    help="Directory where the video is saved.",
)
@click.option(
    "--palette-background",
    "-p",
    "pb_pairs",
    type=click.Tuple([str, ColorType()]),
    multiple=True,
    help="Tuples of the type (palette background)",
)
@click.option(
    "--format",
    default=None,
    type=click.Choice([None, "png", "gif", "mp4", "mov", "webm"]),
    help="Export format for the animation",
)
def render_multiple(
    width: int,
    height: int,
    frame_rate: float,
    video_dir: str,
    format: Literal[None, "png", "gif", "mp4", "mov", "webm"],
    pb_pairs: List[Tuple[str, str]],
):
    video_dir = check_extend_blog_directory(video_dir, "linestar")
    for palette, background in pb_pairs:
        render_linestar(
            width=width,
            height=height,
            frame_rate=frame_rate,
            palette=palette,
            output=None,
            background_color=background,
            video_dir=video_dir,
            format=format,
        )


def check_extend_blog_directory(directory: str | None, folder_name: str):
    if directory is not None:
        support_video_dir = pathlib.Path(directory)
        if str(support_video_dir) == "blog":
            return str(support_video_dir / folder_name)
    return directory


# Entry point for the CLI group
if __name__ == "__main__":
    print(COLORS)
    cli()
