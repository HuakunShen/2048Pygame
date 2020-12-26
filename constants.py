# %%
import pygame

UP = "UP"
DOWN = "DOWN"
LEFT = "LEFT"
RIGHT = "RIGHT"
K_r = "r"
K_q = "q"
# %%
KEY_MAP = {
    pygame.K_r: K_r,
    pygame.K_q: K_q,
    pygame.K_LEFT: LEFT,
    pygame.K_RIGHT: RIGHT,
    pygame.K_UP: UP,
    pygame.K_DOWN: DOWN
}

BG_COLOR_MAP = {
    "BACKGROUND": (172, 157, 142),
    None: (193, 179, 165),
    0: (193, 179, 165),
    1: (233, 221, 209),
    2: (232, 217, 187),
    3: (235, 160, 99),
    4: (237, 128, 78),
    5: (237, 100, 75),
    6: (236, 69, 43),
    7: (231, 197, 90),
    8: (231, 197, 91),
    9: (231, 188, 55),
    10: (230, 185, 38),
    11: (230, 181, 19)
}
