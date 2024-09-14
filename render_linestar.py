from manim import *

from scenes.linestar.linestar import LineStar, LinestarScene
from src.colors import palettes

with tempconfig({"preview": False, "pixel_width":720, "pixel_height":720}):
    print(config.keys())
    scene = LinestarScene()
    scene.curr_palette = palettes.ONEDARK_VIVID_RAINBOW
    scene.background_color = BLACK
    scene.render()