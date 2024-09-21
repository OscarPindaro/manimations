# Manimations

This is a library where I write some mathematical and computational animations.
In the meantime, I will also write some utilities that aim to be reusable between projects.

## Linestar
The base idea is that you can create the illusion of curvature with only straight lines.
I wrote a configurable python script that allows to configure the palette and the background color of a *linestar*.

### Light background
```bash
python render.py render-multiple -w 480 -h 480 -v blog/linestar/light \
-p "ONEDARK_CLASSIC" "EGGSHELL_WHITE"  \
-p "ONEDARK" "EGGSHELL_WHITE" \
-p "GOLDEN_SUNSET" "EGGSHELL_WHITE" \
-p "SUNSET_SKYLINE" "EGGSHELL_WHITE" \
-p "PASTEL_RAINBOW" "EGGSHELL_WHITE" \
-p "HIGH_SATURATION_RAINBOW" "EGGSHELL_WHITE" \
-p "MUTED_RAINBOW" "EGGSHELL_WHITE" \
-p "NEON_RAINBOW" "EGGSHELL_WHITE" \
-p "ONEDARK_CLASS_RAINBOW" "EGGSHELL_WHITE" \
-p "ONEDARK_VIVID_RAINBOW" "EGGSHELL_WHITE" 
```