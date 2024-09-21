import manim
from ..datatypes import NormalizedDict
from typing import Dict

from manim.utils.color import manim_colors

EGGSHELL_WHITE = manim.ManimColor("#F0EAD6")


COLORS: Dict[str, manim.ManimColor | str] = NormalizedDict({
    "EGGSHELL_WHITE":EGGSHELL_WHITE
})

MANIM_COLORS: Dict[str,manim.ManimColor] = NormalizedDict({
    k: v for k, v in vars(manim_colors).items() if isinstance(v, manim.ManimColor)
})