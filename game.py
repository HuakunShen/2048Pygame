from components import *
import pygame
from random import randint, random
from constants import *
import sys


class Game:
    def __init__(self, dimension=4, margin=10, matrix=None):
        self.score = 0
        self.dimension = dimension
        self.matrix = matrix
        self.board = Board(matrix=self.matrix)
        self.width = 800
        self.height = 950
        self.key_down = False
        self.margin = margin
        self.isDone = False
        self.size = self.width, self.height
        self.clock = pygame.time.Clock()
        self.block_size = (self.width - (self.dimension + 1)
                           * self.margin) // self.dimension
        # initialize 2 random tiles
        for _ in range(2):
            self.set_random_tile()
        pygame.init()
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption('2048')

    def draw_grid(self):
        font = pygame.font.Font(None, 64)
        for y in range(len(self.board.get_grid())):
            row = self.board.get_grid()[y]
            for x in range(len(row)):
                tile = row[x]
                rect = pygame.Rect(x * self.block_size + (x + 1) * self.margin, y * self.block_size + (y + 1) *
                                   self.margin,
                                   self.block_size,
                                   self.block_size)
                pygame.draw.rect(self.screen, tile.bg_color, rect)
                if type(tile.value) == int:
                    text_content = str(tile.value)
                else:
                    text_content = ''
                text = font.render(text_content, 30, tile.color)
                textpos = text.get_rect()
                textpos.center = rect.center
                self.screen.blit(text, textpos)

    def update_ui(self):
        self.clock.tick(30)
        self.screen.fill(self.board.bg_color)
        self.draw_grid()
        self.update_score()
        if self.board.is_done():
            self.isDone = True
        self.update_msg()
        pygame.display.flip()

    def main(self):
        while True:
            self.clock.tick(30)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key in KEY_MAP:
                        self.move(KEY_MAP[event.key])
                elif event.type == pygame.KEYUP:
                    self.handle_key_up(event)
            self.screen.fill(self.board.bg_color)
            self.draw_grid()
            self.update_score()
            if self.board.is_done():
                self.isDone = True
            self.update_msg()
            pygame.display.flip()
        pygame.quit()

    def move(self, key):
        self.key_down = True
        score = 0
        changed = False
        if key == K_r:
            print('restart')
            self.__init__(matrix=self.matrix)
            return
        elif key == K_q:
            print('quit game')
            sys.exit(0)
        if not self.isDone:
            score, changed = self.board.move(key)
        self.score += score
        if changed:
            # add a random tile, 2 or 4
            self.set_random_tile()
        return score

    def set_random_tile(self):
        is_4 = random() < 0.1
        target_pos = self.get_new_tile_position()
        self.board.set_tile_power(target_pos, 2 if is_4 else 1)

    def get_new_tile_position(self):
        is_4 = random() < 0.1
        empty_pos = self.board.get_empty_tiles_pos()
        target_pos = empty_pos[randint(0, len(empty_pos) - 1)]
        return target_pos

    def handle_key_up(self, event):
        self.key_down = False

    def update_score(self):
        font = pygame.font.Font(None, 64)
        text_content = 'Score: ' + str(self.score)
        text = font.render(text_content, 30, (255, 255, 255))
        self.screen.blit(text, (50, 820))

    def update_msg(self):
        font = pygame.font.Font(None, 32)
        if self.isDone:
            text_content = 'Game ends, press r to restart'
        else:
            text_content = "Click 'q' to quit the game"
        text = font.render(text_content, 30, (255, 255, 255))
        self.screen.blit(text, (50, 870))


if __name__ == "__main__":
    # m = [
    #     [2, 2, 4, 8],
    #     [2, 2, 4, 8],
    #     [2, 2, 4, 8],
    #     [2, 2, 4, 8],
    # ]
    # game = Game(matrix=m)
    game = Game()
    game.main()
