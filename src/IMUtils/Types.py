from typing import Tuple
from enum import Enum
from dataclasses import dataclass

BBoxYolo = Tuple[int, float, float, float, float]  # (cls, cx, cy, w, h) YOLO format

class Colors(Enum):
    BOX_COLOR = (255, 0, 0)  # Red
    TEXT_COLOR = (255, 255, 255)  # White

@dataclass(frozen=True)
class LetterboxParams:
    ratio: float
    new_size: tuple[int, int]      # (new_w, new_h) after scaling, before padding
    pad: tuple[int, int, int, int] # (left, top, right, bottom)
