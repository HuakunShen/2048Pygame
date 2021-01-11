import sys
import pygame
import numpy as np
from torch import Tensor
from typing import Union
from src.game.controller.game import Game
from src.game.utils import KEY_MAP, K_q, get_bg_color, BG_COLOR


class GameUI(object):
    def __init__(self, matrix: Union[Tensor, np.ndarray] = None, game: Game = None, width: int = 800, height: int = 950,
                 margin: int = 10, fps: int = 30) -> None:
        """
        Init function for GameUI
        :param matrix: predefined 2D matrix for the game, if None then a random board will be generated
        :param game: Game: game object
        :param width: int: game board width
        :param height: int: game board height
        :param margin: margin of game screen
        :param fps: int: target frame per second of animation
        :return: None
        """
        self.fps = fps
        self.game = Game(matrix=matrix) if game is None else game
        self.clock = pygame.time.Clock()
        self.width = width
        self.height = height
        self.margin = margin
        self.block_size = (self.width - (self.game.get_matrix().shape[0] + 1) * margin) // self.game.get_matrix().shape[
            0]
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('2048')

    def run(self) -> None:
        """
        main logic for running the game for playing the game manually
        :return: None
        """
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                elif event.type == pygame.KEYDOWN:
                    if event.key in KEY_MAP:
                        if KEY_MAP[event.key] == K_q:
                            pygame.quit()
                            print("quit 2048")
                            sys.exit(0)
                        else:
                            matrix, score, changed = self.game.move(
                                action=KEY_MAP[event.key], inplace=True)
            self.update_ui()

    def set_game(self, game: Game) -> None:
        """
        Setter for game object
        :param game: Game: game object containing all game state and logic
        :return: None
        """
        self.game = game

    def _update_score(self) -> None:
        """
        update score on UI
        :return: None
        """
        font = pygame.font.Font(None, 64)
        text = font.render(
            'Score: ' + str(self.game.get_score()), 30, (255, 255, 255))
        self.screen.blit(text, (50, 820))

    def _update_msg(self) -> None:
        """
        update messages displayed on UI
        :return: None
        """
        font = pygame.font.Font(None, 32)
        text = font.render('Game ends, press r to restart' if self.game.get_is_done(
        ) else "Click 'q' to quit the game", True, (255, 255, 255))
        self.screen.blit(text, (50, 870))

    def _draw_grid(self) -> None:
        """
        draw the game board grid on UI
        :return: None
        """
        font = pygame.font.Font(None, 64)
        matrix = self.game.get_matrix()
        num_row, num_col = matrix.shape
        for row_i in range(num_row):
            for col_i in range(num_col):
                cell_val = int(matrix[row_i, col_i])
                rect = pygame.Rect(col_i * self.block_size + (col_i + 1) * self.margin,
                                   row_i * self.block_size + (row_i + 1) *
                                   self.margin,
                                   self.block_size,
                                   self.block_size)
                pygame.draw.rect(self.screen, get_bg_color(cell_val), rect)
                text_content = "" if cell_val == 0 else str(cell_val)
                text = font.render(text_content, True, (255, 255, 255))
                text_position = text.get_rect()
                text_position.center = rect.center
                self.screen.blit(text, text_position)

    def update_ui(self) -> None:
        """
        Update UI altogether including updating score, messages displayed and grid
        :return: None
        """
        self.clock.tick(self.fps)
        self.screen.fill(BG_COLOR)
        self._draw_grid()
        self._update_score()
        self._update_msg()
        pygame.display.flip()
