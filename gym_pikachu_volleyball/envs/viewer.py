from __future__ import annotations

import io
import os
import json
import base64
import pygame

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from engine import Engine

import numpy as np

from gym_pikachu_volleyball.envs.constants import *

with open(f'{os.path.dirname(__file__)}/images.json') as f:
    encoded_images: dict = json.load(f)

def get_image_index(state: int, frame_number: int) -> int:
    if state < 4:
        return 5 * state + frame_number
    elif state == 4:
        return 17 + frame_number
    else:
        return 18 + 5 * (state - 5) + frame_number

def load(key: str) -> pygame.surface.Surface:
    return pygame.image.load(io.BytesIO(base64.b64decode(encoded_images[key])))

class Viewer:
    def __init__(self, engine: Engine) -> None:
        pygame.init()

        self.engine = engine
        self.headless = True

        self.screen = pygame.Surface((GROUND_WIDTH, GROUND_HEIGHT))
        self.background = pygame.Surface((GROUND_WIDTH, GROUND_HEIGHT))

        self.clock = pygame.time.Clock()
        self.fps = int(os.environ.get("PIKACHU_FPS", 30))

        self.ball_images = []
        self.player_images = []

        for i in range(5):
            self.ball_images.append(load(f"ball_{i}"))

        for i in range(7):
            for j in range(5):
                if i == 3 and j == 2: break
                if i == 4 and j == 1: break
                self.player_images.append(load(f"pikachu_{i}_{j}"))

        self.ball_hyper_image = load("ball_hyper")
        self.ball_trail_image = load("ball_trail")
        self.ball_punch_image = load("ball_punch")

        self.shadow_image = load("shadow")

        self.ball = engine.ball;

        self.player1 = engine.players[0]
        self.player2 = engine.players[1]

        ground_red_image = load("ground_red")
        ground_yellow_image = load("ground_yellow")

        ground_line_image = load("ground_line")
        ground_line_left_image = load("ground_line_leftmost")
        ground_line_right_image = load("ground_line_rightmost")

        mountain_image = load("mountain")
        sky_blue_image = load("sky_blue")

        net_pillar = load("net_pillar")
        net_pillar_top = load("net_pillar_top")

        for i in range(27):
            self.background.blit(ground_red_image, (16 * i, 248))

        for i in range(27):
            for j in range(2):
                self.background.blit(ground_yellow_image, (16 * i, 280 + 16 * j))

        for i in range(1, 26):
            self.background.blit(ground_line_image, (16 * i, 264))

        self.background.blit(ground_line_left_image, (0, 264))
        self.background.blit(ground_line_right_image, (416, 264))

        self.background.blit(mountain_image, (0, 188))
        
        for i in range(27):
            for j in range(12):
                self.background.blit(sky_blue_image, (16 * i, 16 * j))

        self.background.blit(net_pillar_top, (213, 176))

        for j in range(12):
            self.background.blit(net_pillar, (213, 184 + 8 * j))

    def init_screen(self) -> None:
        pygame.display.init()
        self.screen = pygame.display.set_mode((GROUND_WIDTH, GROUND_HEIGHT))
        self.headless = False

    def update(self) -> None:
        ball_image = self.ball_images[self.ball.rotation]
        player1_image = self.player_images[get_image_index(self.player1.state, self.player1.frame_number)]
        player2_image = self.player_images[get_image_index(self.player2.state, self.player2.frame_number)]

        self.screen.blit(self.background, (0, 0))

        self.screen.blit(self.shadow_image, (self.player1.x - 16, 269))
        self.screen.blit(self.shadow_image, (self.player2.x - 16, 269))
        self.screen.blit(self.shadow_image, (self.ball.x - 16, 269))

        if (self.player1.state == 3 or self.player1.state == 4) and self.player1.diving_direction == -1:
            player1_image = pygame.transform.flip(player1_image, True, False)

        if not ((self.player2.state == 3 or self.player2.state == 4) and self.player2.diving_direction == 1):
            player2_image = pygame.transform.flip(player2_image, True, False)

        self.screen.blit(player1_image, (self.player1.x - 32, self.player1.y - 32))
        self.screen.blit(player2_image, (self.player2.x - 32, self.player2.y - 32))

        if self.ball.is_power_hit:
            self.screen.blit(self.ball_trail_image, (self.ball.previous_previous_x - 20, self.ball.previous_previous_y - 20))
            self.screen.blit(self.ball_hyper_image, (self.ball.previous_x - 20, self.ball.previous_y - 20))

        self.screen.blit(ball_image, (self.ball.x - 20, self.ball.y - 20))

        if self.ball.punch_effect_radius > 0:
            self.ball.punch_effect_radius -= 2
            effect_size = self.ball.punch_effect_radius
            ball_punch_image = pygame.transform.scale(self.ball_punch_image, (effect_size * 2, effect_size * 2))
            self.screen.blit(ball_punch_image, (self.ball.punch_effect_x - effect_size, self.ball.punch_effect_y - effect_size))

    def render(self) -> None:
        if not self.headless:
            self.clock.tick(self.fps)
            pygame.display.update()

    def get_screen_rgb_array(self) -> np.ndarray:
        return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))
    
    def close(self) -> None:
        pygame.quit()
        pygame.display.quit()

