import manim

# palettes
ONEDARK_CLASSIC_PALETTE = [
    "#e5c07b", "#e06c75", "#5c6370", "#f44747", "#56b6c2",
    "#98c379", "#ffffff", "#7f848e", "#abb2bf", "#61afef",
    "#c678dd", "#d19a66", "#BE5046"
]
ONEDARK_VIVID_PALETTE = [
    "#e5c07b", "#ef596f", "#5c6370", "#f44747", "#2bbac5",
    "#89ca78", "#ffffff", "#7f848e", "#abb2bf", "#61afef",
    "#d55fde", "#d19a66", "#BE5046"
]


# some colors
EGGSHELL_WHITE = manim.ManimColor("#F0EAD6")


"""rainbow palettes

The idea of this rainbow palettes is that they can be used to map real values to a rainbow spectrum with:
```python
gradient = color_gradient(colors, 100)
"""
PASTEL_RAINBOW = ["#FFADAD", "#FFD6A5", "#FDFFBF", "#CAFFBF",  "#A0C4FF", ]
HIGH_SATURATION_RAINBOW = ["#ff0000","#fc4444","#fc6404","#fcd444","#8cc43c","#029658","#1abc9c","#5bc0de","#6454ac"]
MUTED_RAINBOW = ["#d46a6c,#e9c377","#89a35f","#755f91","#a64d99",]
NEON_RAINBOW = ['#ff0000', '#ff7700','#ffff00', '#00ff00', '#0000ff', '#7700ff', '#ee00ee']
ONEDARK_CLASS_RAINBOW = ["#f44747", "#d19a66",  "#e5c07b","#98c379", "#61afef", "#c678dd"]
ONEDARK_VIVID_RAINBOW = ["#f44747", "#d19a66",  "#e5c07b","#89ca78", "#61afef",  "#d55fde"]
