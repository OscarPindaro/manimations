from manim import Scene, Circle, Square, Triangle, Transform, Arrow
from manim import *
from manimations.colors import OneDarkClassicPalette


class TransformCycle(Scene):
    def construct(self):
        a = Arrow(start=ORIGIN, end=UP, color=OneDarkClassicPalette.RED)
        self.add(a)
        self.wait()
