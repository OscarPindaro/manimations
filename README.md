# Manimations

This is a library where I write some mathematical and computational animations.
In the meantime, I will also write some utilities that aim to be reusable between projects.

## Linestar
The base idea is that you can create the illusion of curvature with only straight lines.
I wrote a configurable python script that allows to configure the palette and the background color of a *linestar*.

### Light background
```bash
python render.py render-multiple -w 720 -h 720 --format webm -v blog/linestar/light \
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

### Dark background
```bash
python render.py render-multiple -w 720 -h 720 --format gif -v blog/linestar/dark \
-p "ONEDARK_CLASSIC" "#282c33"  \
-p "ONEDARK" "#282c33" \
-p "GOLDEN_SUNSET" "#282c33" \
-p "SUNSET_SKYLINE" "#282c33" \
-p "PASTEL_RAINBOW" "#282c33" \
-p "HIGH_SATURATION_RAINBOW" "#282c33" \
-p "MUTED_RAINBOW" "#282c33" \
-p "NEON_RAINBOW" "#282c33" \
-p "ONEDARK_CLASS_RAINBOW" "#282c33" \
-p "ONEDARK_VIVID_RAINBOW" "#282c33" 
```