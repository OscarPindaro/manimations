import click
from manim import *
from scenes.linestar.linestar import LineStar, LinestarScene
from src.colors import PALETTES, COLORS, MANIM_COLORS
import re


# Create a custom Click type for hex color or named color validation
class ColorType(click.ParamType):
    name = "color"

    def convert(self, value, param, ctx):
        # Check if the value is a valid hex color
        if re.match(r"^#([A-Fa-f0-9]{6})$", value):
            return value
        # Check if the value exists in all_colors or all_manim_colors
        elif value in COLORS or value in MANIM_COLORS:
            return value
        else:
            self.fail(
                f"'{value}' is not a valid hex color or known color. Expected a hex format like '#FFFFFF' or a known color name from COLORS or MANIM_COLORS."
            )


# Create a custom palette validator callback
def validate_palette(ctx, param, value):
    if value in PALETTES:
        return value
    else:
        raise click.BadParameter(
            f"'{value}' is not a valid palette. Choose from: {list(PALETTES.keys())}"
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
@click.option("--video-dir", "-v", type=str, help="Directory where the video is saved")
def render_linestar(
    width: int,
    height: int,
    frame_rate: float,
    output: str,
    palette: str,
    background_color: str,
    video_dir: str,
):

    # Generate the default output file name if none is provided
    if not output:
        # Normalize the palette and background color for the filename
        normalized_palette = palette.lower().replace(" ", "_")
        normalized_background = (
            background_color.lstrip("#").lower().replace(" ", "_")
        )  # If hex, strip the '#'
        scene_class_name = LinestarScene.__name__
        output = f"{scene_class_name}_{normalized_palette}_{normalized_background}.mp4"

    # Proceed with rendering the scene
    config = {
        "preview": False,
        "pixel_width": width,
        "pixel_height": height,
        "output_file": output,
        "frame_rate": frame_rate,
    }

    if video_dir is not None:
        config["video_dir"] = video_dir
    with tempconfig(config):
        print(config.keys())  # Debug print
        scene = LinestarScene()
        scene.curr_palette = PALETTES[palette]  # Use the validated palette
        scene.background_color = background_color  # Set the validated background color
        scene.render()


# Entry point for the CLI group
if __name__ == "__main__":
    cli()
