import pygame
from gamestate import GameState
from components import KEY_MAP


class Game(object):
    def __init__(self, dimension=4, score=0, init_matrix=None, margin=10, width=800, height=950, game_state=None):
        self.game_state = game_state if game_state is not None else GameState(dimension=dimension, score=score,
                                                                              matrix=init_matrix)
        self.width = width
        self.height = height
        self.init_matrix = init_matrix
        self.clock = pygame.time.Clock()
        self.margin = margin
        self.key_down = False
        self.block_size = (self.width - (dimension + 1)
                           * self.margin) // dimension
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('2048')

    def set_game_state(self, game_state: GameState):
        self.game_state = game_state.copy()

    def is_done(self):
        return self.game_state.get_is_done()

    def handle_key_up(self):
        self.key_down = False

    def update_score(self):
        font = pygame.font.Font(None, 64)
        text_content = 'Score: ' + str(self.game_state.get_score())
        text = font.render(text_content, 30, (255, 255, 255))
        self.screen.blit(text, (50, 820))

    def update_msg(self):
        font = pygame.font.Font(None, 32)
        text_content = 'Game ends, press r to restart' if self.is_done(
        ) else "Click 'q' to quit the game"
        text = font.render(text_content, True, (255, 255, 255))
        self.screen.blit(text, (50, 870))

    def draw_grid(self):
        font = pygame.font.Font(None, 64)
        for y in range(len(self.game_state.get_board().get_grid())):
            row = self.game_state.get_board().get_grid()[y]
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
                text = font.render(text_content, True, tile.color)
                text_position = text.get_rect()
                text_position.center = rect.center
                self.screen.blit(text, text_position)

    def update_ui(self):
        self.clock.tick(30)
        self.screen.fill(self.game_state.get_board().bg_color)
        self.draw_grid()
        self.update_score()
        self.update_msg()
        pygame.display.flip()

    def main(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                elif event.type == pygame.KEYDOWN:
                    if event.key in KEY_MAP:
                        self.game_state.move(KEY_MAP[event.key])
                elif event.type == pygame.KEYUP:
                    self.handle_key_up()
            self.update_ui()


if __name__ == '__main__':
    game = Game()
    game.main()
