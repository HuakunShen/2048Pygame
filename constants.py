import pygame

SEED = 2048

DEFAULT_GOAL = 2048

UP = "UP"
DOWN = "DOWN"
LEFT = "LEFT"
RIGHT = "RIGHT"

K_r = "r"
K_q = "q"

ARROW_KEYS = [UP, DOWN, LEFT, RIGHT]


KEY_MAP = {
    pygame.K_r: K_r,
    pygame.K_q: K_q,
    pygame.K_LEFT: LEFT,
    pygame.K_RIGHT: RIGHT,
    pygame.K_UP: UP,
    pygame.K_DOWN: DOWN
}

BG_COLOR = (172, 157, 142)

CELL_BG_COLOR_MAP = {
    0: (193, 179, 165),
    2: (233, 221, 209),
    4: (232, 217, 187),
    8: (235, 160, 99),
    16: (237, 128, 78),
    32: (237, 100, 75),
    64: (236, 69, 43),
    128: (231, 197, 90),
    256: (231, 197, 91),
    512: (231, 188, 55),
    1024: (230, 185, 38),
    2048: (230, 181, 19)
}
