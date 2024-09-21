import click
from manim import *
from scenes.linestar.linestar import LineStar, LinestarScene
from src.colors import palettes
import re

# Create a custom Click type for hex color validation
class HexColorType(click.ParamType):
    name = 'hexcolor'

    def convert(self, value, param, ctx):
        if re.match(r'^#([A-Fa-f0-9]{6})$', value):
            return ManimColor(value)
        else:
            self.fail(f"{value} is not a valid hex color. Expected format: '#FFFFFF' or '#ffffff'")

# Create a custom palette validator callback
def validate_palette(ctx, param, value):
    if value in palettes:
        return value
    else:
        raise click.BadParameter(f"'{value}' is not a valid palette. Choose from: {list(palettes.keys())}")

# Create the CLI group
@click.group()
def cli():
    pass

# Add a 'render' command to the CLI
@cli.command("linestar")
@click.option('--width', '-w', default=720, type=int, help="Width of the output video.")
@click.option('--height', '-h', default=720, type=int, help="Height of the output video.")
@click.option('--output', '-o', default="output.mp4", type=str, help="Output file name.")
@click.option('--palette', '-p', required=True, callback=validate_palette, help="Color palette to use.")
@click.option('--background-color', '-b', required=True, type=HexColorType(), help="Background color in hex format.")
def render_linestar(width: int, height: int, output: str, palette: str, background_color: str):
    
    # At this point, both palette and background_color are already validated
    with tempconfig({"preview": False, "pixel_width": width, "pixel_height": height, "output_file": output}):
        scene = LinestarScene()
        scene.curr_palette = palettes[palette]  # Use the validated palette
        print(type(background_color))
        scene.background_color = background_color  # Set background color
        scene.render()

# Entry point for the CLI group
if __name__ == "__main__":
    cli()
