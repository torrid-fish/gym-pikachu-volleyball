import random
from typing import Tuple
import numpy as np
import cv2

from gym_pikachu_volleyball.envs.viewer import Viewer
from gym_pikachu_volleyball.envs.constants import *
from gym_pikachu_volleyball.envs.common import UserInput

class Engine:
    __slots__ = ['players', 'ball', 'viewer', 'more_random']

    def __init__(self, more_random: bool) -> None:
        self.players = (
                Player(False), 
                Player(True))

        self.ball = Ball(False)
        self.more_random = more_random
        
    def step(self, user_inputs: Tuple[UserInput, UserInput]) -> bool:
        is_ball_touching_ground =\
                self.__process_collision_between_ball_and_world_and_set_ball_position();

        for i in range(2):
            self.__calculate_expected_landing_point_x_for(self.ball)
            self.__process_player_movement_and_set_player_position(i, user_inputs[i])

        for i in range(2):
            is_happening = self.__is_collision_between_ball_and_player_happening(i)
            player = self.players[i]
            if is_happening:
                if not player.is_collision_with_ball_happening:
                    self.__process_collision_between_ball_and_player(i, user_inputs[i])
                    player.is_collision_with_ball_happening = True
            else:
                player.is_collision_with_ball_happening = False

        return is_ball_touching_ground

    def reset(self, is_player2_serve: bool) -> None:
        self.players[0].reset()
        self.players[1].reset()
        self.ball.reset(is_player2_serve, self.more_random)

    def seed(self, seed: int) -> None:
        random.seed(seed)

    def __is_collision_between_ball_and_player_happening(self, player_id: int) -> bool:
        player = self.players[player_id]
        return abs(self.ball.x - player.x) <= PLAYER_HALF_LENGTH and\
                abs(self.ball.y - player.y) <= PLAYER_HALF_LENGTH

    def __process_collision_between_ball_and_world_and_set_ball_position(self) -> bool:
        self.ball.previous_previous_x = self.ball.previous_x
        self.ball.previous_previous_y = self.ball.previous_y
        self.ball.previous_x = self.ball.x
        self.ball.previous_y = self.ball.y

        self.ball.fine_rotation = (self.ball.fine_rotation + self.ball.x_velocity // 2) % 50
        self.ball.rotation = self.ball.fine_rotation // 10

        future_ball_x = self.ball.x + self.ball.x_velocity

        if future_ball_x < BALL_RADIUS or future_ball_x > GROUND_WIDTH:
            self.ball.x_velocity = -self.ball.x_velocity

        future_ball_y = self.ball.y + self.ball.y_velocity
        if future_ball_y < 0:
            self.ball.y_velocity = 1

        if abs(self.ball.x - GROUND_HALF_WIDTH) < NET_PILLAR_HALF_WIDTH and\
                self.ball.y > NET_PILLAR_TOP_TOP_Y_COORD:
            if self.ball.y <= NET_PILLAR_TOP_BOTTOM_Y_COORD:
                if self.ball.y_velocity > 0:
                    self.ball.y_velocity = -self.ball.y_velocity
            else:
                if self.ball.x < GROUND_HALF_WIDTH:
                    self.ball.x_velocity = -abs(self.ball.x_velocity)
                else:
                    self.ball.x_velocity = abs(self.ball.x_velocity)

        future_ball_y = self.ball.y + self.ball.y_velocity

        if future_ball_y > BALL_TOUCHING_GROUND_Y_COORD:
            self.ball.y = BALL_TOUCHING_GROUND_Y_COORD
            self.ball.y_velocity = -self.ball.y_velocity
            self.ball.punch_effect_x = self.ball.x
            self.ball.punch_effect_y = BALL_TOUCHING_GROUND_Y_COORD + BALL_RADIUS
            self.ball.punch_effect_radius = BALL_RADIUS
            return True

        self.ball.y = future_ball_y
        self.ball.x = self.ball.x + self.ball.x_velocity
        self.ball.y_velocity += 1

        return False

    def __process_player_movement_and_set_player_position(self, player_id: int, user_input: UserInput):
        player = self.players[player_id]

        if player.state == 4:
            player.lying_down_duration_left -= 1
            if player.lying_down_duration_left < -1:
                player.state = 0
            return

        player_velocity_x = 0
        if player.state < 5:
            if player.state < 3:
                player_velocity_x = user_input.x_direction * 6
            else:
                player_velocity_x = player.diving_direction * 8

        future_player_x = player.x + player_velocity_x
        player.x = future_player_x

        if not player.is_player2:
            if future_player_x < PLAYER_HALF_LENGTH:
                player.x = PLAYER_HALF_LENGTH
            elif future_player_x > GROUND_HALF_WIDTH - PLAYER_HALF_LENGTH:
                player.x = GROUND_HALF_WIDTH - PLAYER_HALF_LENGTH
        else:
            if future_player_x < GROUND_HALF_WIDTH + PLAYER_HALF_LENGTH:
                player.x = GROUND_HALF_WIDTH + PLAYER_HALF_LENGTH
            elif future_player_x > GROUND_WIDTH - PLAYER_HALF_LENGTH:
                player.x = GROUND_WIDTH - PLAYER_HALF_LENGTH

        if player.state < 3 and user_input.y_direction == -1 and\
                player.y == PLAYER_TOUCHING_GROUND_Y_COORD:
            player.y_velocity = -16
            player.state = 1
            player.frame_number = 0

        future_player_y = player.y + player.y_velocity
        player.y = future_player_y

        if future_player_y < PLAYER_TOUCHING_GROUND_Y_COORD:
            player.y_velocity += 1
        elif future_player_y > PLAYER_TOUCHING_GROUND_Y_COORD:
            player.y_velocity = 0
            player.y = PLAYER_TOUCHING_GROUND_Y_COORD
            player.frame_number = 0

            if player.state == 3:
                player.state = 4
                player.frame_number = 0
                player.lying_down_duration_left = 3
            else:
                player.state = 0

        if user_input.power_hit == 1:
            if player.state == 1:
                player.delay_before_next_frame = 5
                player.frame_number = 0
                player.state = 2
            elif player.state == 0 and user_input.x_direction != 0:
                player.state = 3
                player.frame_number = 0
                player.diving_direction = user_input.x_direction
                player.y_velocity = -5

        if player.state == 1:
            player.frame_number = (player.frame_number + 1) % 3
        elif player.state == 2:
            if player.delay_before_next_frame < 1:
                player.frame_number += 1
                if player.frame_number > 4:
                    player.frame_number = 0
                    player.state = 1
            else:
                player.delay_before_next_frame += 1
        elif player.state == 0:
            player.delay_before_next_frame += 1
            if player.delay_before_next_frame > 3:
                player.delay_before_next_frame = 0
                future_frame_number = player.frame_number + player.normal_status_arm_swing_direction
                if future_frame_number < 0 or future_frame_number > 4:
                    player.normal_status_arm_swing_direction *= -1
                player.frame_number += player.normal_status_arm_swing_direction

        if player.game_ended:
            if player.state == 0:
                if player.is_winner:
                    player.state = 5
                else:
                    player.state = 6
                player.delay_before_next_frame = 0
                player.frame_number = 0

            self.__process_game_end_frame_for(player_id)

    def __process_collision_between_ball_and_player(self, player_id: int, user_input: UserInput):
        player = self.players[player_id]

        if self.ball.x < player.x:
            self.ball.x_velocity = -abs(self.ball.x - player.x) // 3
        elif self.ball.x > player.x:
            self.ball.x_velocity = abs(self.ball.x - player.x) // 3

        if self.ball.x_velocity == 0:
            self.ball.x_velocity = random.randint(-1, +1)

        ball_abs_y_velocity = abs(self.ball.y_velocity)
        self.ball.y_velocity = -ball_abs_y_velocity

        if ball_abs_y_velocity < 15:
            self.ball.y_velocity = -15

        if player.state == 2:
            if self.ball.x < GROUND_HALF_WIDTH:
                self.ball.x_velocity = (abs(user_input.x_direction) + 1) * 10
            else:
                self.ball.x_velocity = -(abs(user_input.x_direction) + 1) * 10
            
            self.ball.punch_effect_x = self.ball.x
            self.ball.punch_effect_y = self.ball.y

            self.ball.y_velocity = abs(self.ball.y_velocity) * user_input.y_direction * 2
            self.ball.punch_effect_radius = BALL_RADIUS

            self.ball.is_power_hit = True
        else:
            self.ball.is_power_hit = False

    def __process_game_end_frame_for(self, player_id: int) -> None:
        player = self.players[player_id]
        if player.game_ended and player.frame_number < 4:
            player.delay_before_next_frame += 1
            if player.delay_before_next_frame > 4:
                player.delay_before_next_frame = 0
                player.frame_number += 1

    def get_obs(self, pixel_mode):
        if pixel_mode:
            return self.viewer.get_screen_rgb_array()
        else:
            ret = []
            for player in self.players:
                ret.append(player.x)
                ret.append(player.y)
                ret.append(player.y_velocity)
            ret.append(self.ball.x)
            ret.append(self.ball.y)
            ret.append(self.ball.x_velocity)
            ret.append(self.ball.y_velocity)
            return np.array(ret, dtype=np.float32)
    
    def get_other_obs(self, pixel_mode):
        if pixel_mode:
            return cv2.flip(self.viewer.get_screen_rgb_array(), 1) # horizontal flip
        else:
            ret = []
            for player in self.players[::-1]: 
                ret.append(GROUND_WIDTH - player.x)
                ret.append(player.y)
                ret.append(player.y_velocity)
            ret.append(GROUND_WIDTH - self.ball.x)
            ret.append(self.ball.y)
            ret.append(-self.ball.x_velocity)
            ret.append(self.ball.y_velocity)
            return np.array(ret, dtype=np.float32)

    def create_viewer(self, render_mode: str) -> None:
        self.viewer = Viewer(self)
        if render_mode == 'human':
            self.viewer.init_screen()

    def render(self, mode: str) -> None:
        if mode == "human":
            self.viewer.render()
        else:
            return self.viewer.get_screen_rgb_array() 
        
    def __expected_landing_point_x_when_power_hit(self, userInputXDirection: int, userInputYDirection: int, ball) -> int:
        copy_ball = CopyBall(ball.x, ball.y, ball.x_velocity, ball.y_velocity)
        
        if copy_ball.x < GROUND_HALF_WIDTH:
            copy_ball.x_velocity = (abs(userInputXDirection) + 1) * 10
        else:
            copy_ball.x_velocity = -(abs(userInputXDirection) + 1) * 10

        copy_ball.y_velocity = abs(copy_ball.y_velocity) * userInputYDirection * 2
        loopCounter: int = 0
        while True:
            loopCounter += 1

            futurecopy_ballX = copy_ball.x + copy_ball.x_velocity
            if futurecopy_ballX < BALL_RADIUS or futurecopy_ballX > GROUND_WIDTH:
                copy_ball.x_velocity = -copy_ball.x_velocity

            if copy_ball.y + copy_ball.y_velocity < 0:
                copy_ball.y_velocity = 1

            if abs(copy_ball.x - GROUND_HALF_WIDTH) < NET_PILLAR_HALF_WIDTH and copy_ball.y > NET_PILLAR_TOP_TOP_Y_COORD:
                if copy_ball.y <= NET_PILLAR_TOP_BOTTOM_Y_COORD:
                    if copy_ball.y_velocity > 0:
                        copy_ball.y_velocity = -copy_ball.y_velocity
                else:
                    if copy_ball.x < GROUND_HALF_WIDTH:
                        copy_ball.x_velocity = -abs(copy_ball.x_velocity)
                    else:
                        copy_ball.x_velocity = abs(copy_ball.x_velocity)

            copy_ball.y = copy_ball.y + copy_ball.y_velocity

            if copy_ball.y > BALL_TOUCHING_GROUND_Y_COORD or loopCounter >= 1000:
                return copy_ball.x

            copy_ball.x = copy_ball.x + copy_ball.x_velocity
            copy_ball.y_velocity += 1
 
    def __decide_wheter_input_power_hit(self, player, ball, theOtherPlayer, userInput) -> bool:
        if random.randrange(0, 2) == 0:
            for xDirection in range(1, -1, -1):
                for yDirection in range(-1, 2):
                    expectedLandingPointX = self.__expected_landing_point_x_when_power_hit(xDirection, yDirection, ball)
                    if (expectedLandingPointX <= int(player.is_player2) * GROUND_HALF_WIDTH or\
                        expectedLandingPointX >= int(player.is_player2) * GROUND_WIDTH + GROUND_HALF_WIDTH) and\
                        abs(expectedLandingPointX - theOtherPlayer.x) > PLAYER_LENGTH:
                            userInput.x_direction = xDirection
                            userInput.y_direction = yDirection
                            return True
        else:
            for xDirection in range(1, -1, -1):
                for yDirection in range(1, -2, -1):
                    expectedLandingPointX = self.__expected_landing_point_x_when_power_hit(xDirection, yDirection, ball)
                    if (expectedLandingPointX <= int(player.is_player2) * GROUND_HALF_WIDTH or\
                        expectedLandingPointX >= int(player.is_player2) * GROUND_WIDTH + GROUND_HALF_WIDTH) and\
                        abs(expectedLandingPointX - theOtherPlayer.x) > PLAYER_LENGTH:
                            userInput.x_direction = xDirection
                            userInput.y_direction = yDirection
                            return True
        return False

    def let_computer_decide_user_input(self, player_id):
        ball = self.ball
        player = self.players[player_id]
        the_other_player = self.players[1 - player_id]
        user_input = UserInput()

        virtual_expected_landing_point_x: int = ball.expected_landing_point_x

        if abs(ball.x - player.x) > 100 and abs(ball.x_velocity) < player.computer_boldness + 5:
            leftBoundary: int = int(player.is_player2) * GROUND_HALF_WIDTH
            if (ball.expected_landing_point_x <= leftBoundary or\
            ball.expected_landing_point_x >= GROUND_WIDTH + GROUND_HALF_WIDTH) and\
            player.computer_where_to_stand_by == 0:
                virtual_expected_landing_point_x = leftBoundary + GROUND_HALF_WIDTH // 2

        if abs(virtual_expected_landing_point_x - player.x) > player.computer_boldness + 8:
            user_input.x_direction = 1 if player.x < virtual_expected_landing_point_x else -1

        elif random.randrange(0, 20) == 0:
            player.computer_where_to_stand_by = random.randrange(0, 2)

        if player.state == 0:
            if abs(ball.x_velocity) < player.computer_boldness + 3 and\
            abs(ball.x - player.x) < PLAYER_HALF_LENGTH and\
            ball.y > -36 and ball.y < 10 * player.computer_boldness + 84 and ball.y_velocity > 0:
                user_input.y_direction = -1
            
            leftBoundary: int = int(player.is_player2) * GROUND_HALF_WIDTH
            rightBoundary: int = (int(player.is_player2) + 1) * GROUND_HALF_WIDTH

            if ball.expected_landing_point_x > leftBoundary and ball.expected_landing_point_x < rightBoundary and\
            abs(ball.x - player.x) > player.computer_boldness * 5 + PLAYER_LENGTH and\
            ball.x > leftBoundary and ball.x < rightBoundary and ball.y > 174:
                user_input.power_hit = 1
                user_input.x_direction = 1 if player.x < ball.x else -1

        elif player.state == 1 or player.state == 2:
            if abs(ball.x - player.x) > 8:
                user_input.x_direction = 1 if player.x < ball.x else -1

            if abs(ball.x - player.x) < 48 and abs(ball.y - player.y) < 48:
                willInputPowerHit: bool = self.__decide_wheter_input_power_hit(player, ball, the_other_player, user_input)
                
                if willInputPowerHit:
                    user_input.power_hit = 1
                    if abs(the_other_player.x - player.x) < 80 and user_input.y_direction != -1:
                        user_input.y_direction = -1
        return user_input    

    def __calculate_expected_landing_point_x_for(self, ball):
        copy_ball = CopyBall(ball.x, ball.y, ball.x_velocity, ball.y_velocity)
        
        loopCounter: int = 0
        while True:
            loopCounter += 1

            future_copy_ball_x: int = copy_ball.x_velocity + copy_ball.x
            if future_copy_ball_x < BALL_RADIUS or future_copy_ball_x > GROUND_WIDTH:
                copy_ball.x_velocity = -copy_ball.x_velocity
            if copy_ball.y + copy_ball.y_velocity < 0:
                copy_ball.y_velocity = 1

            if abs(copy_ball.x - GROUND_HALF_WIDTH) < NET_PILLAR_HALF_WIDTH and copy_ball.y > NET_PILLAR_TOP_TOP_Y_COORD:
                if copy_ball.y < NET_PILLAR_TOP_BOTTOM_Y_COORD:
                    if copy_ball.y_velocity > 0:
                        copy_ball.y_velocity = -copy_ball.y_velocity
                else:
                    if copy_ball.x < GROUND_HALF_WIDTH:
                        copy_ball.x_velocity = -abs(copy_ball.x_velocity)
                    else:
                        copy_ball.x_velocity = abs(copy_ball.x_velocity)
            
            copy_ball.y = copy_ball.y + copy_ball.y_velocity

            if copy_ball.y > BALL_TOUCHING_GROUND_Y_COORD or loopCounter >= 1000:
                break

            copy_ball.x = copy_ball.x + copy_ball.x_velocity
            copy_ball.y_velocity += 1

        ball.expected_landing_point_x = copy_ball.x

    def close(self) -> None:
        self.viewer.close()

class Player:
    __slots__ = ['is_player2', 
            'diving_direction', 'lying_down_duration_left', 
            'is_winner', 'game_ended', 'computer_where_to_stand_by', 
            'x', 'y', 'y_velocity', 'is_collision_with_ball_happening', 
            'state', 'frame_number', 'normal_status_arm_swing_direction', 
            'delay_before_next_frame', 'computer_boldness']

    def __init__(self, is_player2: bool) -> None:
        self.is_player2 = is_player2

        self.diving_direction = 0
        self.lying_down_duration_left = -1
        self.is_winner = False
        self.game_ended = False

        self.computer_where_to_stand_by = 0

        self.reset()

    def reset(self) -> None:
        self.x = 36 if not self.is_player2 else GROUND_WIDTH - 36
        self.y = PLAYER_TOUCHING_GROUND_Y_COORD

        self.y_velocity = 0
        self.is_collision_with_ball_happening = False

        self.state = 0
        self.frame_number = 0
        self.normal_status_arm_swing_direction = 1
        self.delay_before_next_frame = 0

        self.computer_boldness = random.randrange(0, 5)

class Ball:
    __slots__ = ['expected_landing_point_x', 
            'rotation', 'fine_rotation', 
            'punch_effect_x', 'punch_effect_y', 
            'previous_x', 'previous_previous_x', 
            'previous_y', 'previous_previous_y', 
            'x', 'y', 'x_velocity', 'y_velocity', 
            'punch_effect_radius', 'is_power_hit']
    
    def __init__(self, is_player2_serve: bool) -> None:
        self.reset(is_player2_serve, False)

        self.expected_landing_point_x = 0

        self.rotation = 0
        self.fine_rotation = 0
        self.punch_effect_x = 0
        self.punch_effect_y = 0

        self.previous_x = 0
        self.previous_previous_x = 0
        self.previous_y = 0
        self.previous_previous_y = 0

    def reset(self, is_player2_serve: bool, more_random: bool) -> None:
        self.x = 56 if not is_player2_serve else GROUND_WIDTH - 56
        self.y = 0

        self.x_velocity = 0
        # add randomness
        if more_random:
            self.x = GROUND_HALF_WIDTH
            self.x_velocity = np.random.randint(low=-20, high=20)
            self.y_velocity = np.random.randint(low=-10, high=0)
        else:
            self.y_velocity = 1

        self.punch_effect_radius = 0
        self.is_power_hit = False

class CopyBall:
    __slots__ = ['x', 'y', 'x_velocity', 'y_velocity']

    def __init__(self, x, y, x_velocity, y_velocity) -> None:
        self.x = x
        self.y = y
        self.x_velocity = x_velocity
        self.y_velocity = y_velocity