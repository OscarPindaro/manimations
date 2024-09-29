from typing import List


class OneDarkClassicPalette:
    YELLOW: str = "#e5c07b"
    PINK: str = "#e06c75"
    GRAY_C: str = "#5c6370"
    RED: str = "#f44747"
    LIGHT_BLUE: str = "#56b6c2"
    GREEN: str = "#98c379"
    WHITE: str = "#ffffff"
    GRAY_B: str = "#7f848e"
    GRAY_A: str = "#abb2bf"
    BLUE: str = "#61afef"
    PURPLE: str = "#c678dd"
    ORANGE: str = "#d19a66"
    BRICK_RED: str = "#BE5046"

    # Aliases for grays
    DARK_GRAY: str = GRAY_C
    MEDIUM_GRAY: str = GRAY_B
    LIGHT_GRAY: str = GRAY_A


class OneDarkVividPalette:
    YELLOW: str = "#e5c07b"
    PINK: str = "#ef596f"
    GRAY_C: str = "#5c6370"
    RED: str = "#f44747"
    LIGHT_BLUE: str = "#2bbac5"
    GREEN: str = "#89ca78"
    WHITE: str = "#ffffff"
    GRAY_B: str = "#7f848e"
    GRAY_A: str = "#abb2bf"
    BLUE: str = "#61afef"
    PURPLE: str = "#d55fde"
    ORANGE: str = "#d19a66"
    BRICK_RED: str = "#BE5046"

    # Aliases for grays
    DARK_GRAY: str = GRAY_C
    MEDIUM_GRAY: str = GRAY_B
    LIGHT_GRAY: str = GRAY_A


class GoldenSunsetPalette:
    YELLOW_B: str = "#fdb813"
    YELLOW_A: str = "#ffc300"
    ORANGE: str = "#ff5733"
    RED_A: str = "#c70039"
    RED_B: str = "#900c3f"

    LIGHT_YELLOW: str = YELLOW_A
    DARK_YELLOW: str = YELLOW_B

    LIGHT_RED: str = RED_A
    DARK_RED: str = RED_B

    ORDERED_PALETTE: List[str] = [YELLOW_B, YELLOW_A, ORANGE, RED_A, RED_B]


class SunsetSkylinePalette:
    YELLOW_A: str = "#ffdfb3"
    YELLOW_B: str = "#faad45"
    ORANGE: str = "#f46046"
    RED: str = "#bc0967"
    PURPLE: str = "#411259"

    LIGHT_YELLOW: str = YELLOW_A
    DARK_YELLOW: str = YELLOW_B

    ORDERED_PALETTE: List[str] = [YELLOW_A, YELLOW_B, ORANGE, RED, PURPLE]
