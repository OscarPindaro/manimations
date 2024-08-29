from manim import *
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

class VeryBasicLineStar(Scene):
    
    N_SAMPLES: int = 100
    Y_RANGE: Tuple[int,int] = (-3.5,3.5)
    X_RANGE: Tuple[int,int] = (-3.5,3.5)
    
    def construct(self):
        
        
        assert self.N_SAMPLES % 2 == 0, "N_SAMPLE should be even"
        
        x_axis, y_axis = get_x_y_coords(self.X_RANGE, self.Y_RANGE, self.N_SAMPLES)
        assert y_axis.shape == x_axis.shape
        lines: List[Line]  = []
        for x, y in zip(x_axis,y_axis):
            lines.append(Line((0,y,0), (x,0,0)))

        self.play(Create(line) for line in lines)
        
        
class LaggedLineStar(Scene):
    
    N_SAMPLES: int = 20
    Y_RANGE: Tuple[int,int] = (-3.5,3.5)
    X_RANGE: Tuple[int,int] = (-3.5,3.5)
    
    def construct(self):
        
        
        assert self.N_SAMPLES % 2 == 0, "N_SAMPLE should be even"
        
        x_axis, y_axis = get_x_y_coords(self.X_RANGE, self.Y_RANGE, self.N_SAMPLES)
        assert y_axis.shape == x_axis.shape
        lines: List[Line]  = []
        for x, y in zip(x_axis,y_axis):
            lines.append(Line((0,y,0), (x,0,0)))

        self.play(LaggedStart([Create(line) for line in lines]))
        
        
def get_x_y_coords(x_range: Tuple[int,int], y_range: Tuple[int,int], n_samples: int) -> Tuple[np.ndarray, np.ndarray]: 
        y_axis = np.concatenate(
            (
            np.linspace(y_range[1],y_range[0], n_samples, endpoint=True), 
            np.linspace(y_range[0],y_range[1], n_samples, endpoint=True)
            )
        )
        x_axis = np.concatenate(
            (
                np.linspace(0,x_range[0],n_samples//2, endpoint=True),
                np.linspace(x_range[0],0,n_samples//2, endpoint=True),
                np.linspace(0,x_range[1],n_samples//2, endpoint=True),
                np.linspace(x_range[1],0,n_samples//2, endpoint=True),
            )
        )
        return x_axis, y_axis
        

class ColoredLineStar(Scene):
    
    N_SAMPLES: int = 100
    Y_RANGE: Tuple[int,int] = (-3.5,3.5)
    X_RANGE: Tuple[int,int] = (-3.5,3.5)
    PALETTE_0 = color_gradient([PURPLE, WHITE, YELLOW], 2*N_SAMPLES)
    
    def construct(self):
        
        
        assert self.N_SAMPLES % 2 == 0, "N_SAMPLE should be even"
        
        x_axis, y_axis = get_x_y_coords(self.X_RANGE, self.Y_RANGE, self.N_SAMPLES)
        assert y_axis.shape == x_axis.shape
        lines: List[Line]  = []
        
        for x, y, color in zip(x_axis,y_axis, self.PALETTE_0):
            lines.append(Line((0,y,0), (x,0,0), color=color))

        self.play(Create(line) for line in lines)
        
Square()
Circle()

@dataclass
class LineStar:
    x_range: Tuple[int,int]   
    y_range: Tuple[int,int] 
    axis_resolution: int


config.pixel_width = config.pixel_height
config.frame_width = config.frame_height
        