from abc import ABC, abstractmethod
from typing import Union
from constants import UP, DOWN, LEFT, RIGHT


class Player(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_move(self) -> Union[UP, DOWN, LEFT, RIGHT]:
        raise NotImplemented
