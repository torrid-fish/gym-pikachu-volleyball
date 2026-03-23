import gymnasium as gym
import numpy as np

from gymnasium import spaces
from typing import Tuple, Union, Optional

from gym_pikachu_volleyball.envs.engine import Engine
from gym_pikachu_volleyball.envs.common import convert_to_user_input
from gym_pikachu_volleyball.envs.constants import GROUND_HALF_WIDTH

class PikachuVolleyballEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}
    more_random = False
    pixel_mode = False
    is_player2_serve = False

    def __init__(self, render_mode: str, limited_timestep: int):
        super(PikachuVolleyballEnv, self).__init__()

        self.action_space = spaces.Discrete(18)
        
        if self.pixel_mode:
            observation_size = (304, 432, 3)
            self.observation_space = spaces.Box(
                    low = np.zeros(observation_size), 
                    high = 255 * np.ones(observation_size),
                    dtype = np.uint8)
        else:
            observation_size = 10
            high = np.array([np.finfo(np.float32).max] * 10)
            self.observation_space = spaces.Box(-high, high)

        self.engine = Engine(self.more_random)
        self.engine.create_viewer(render_mode)

        self.render_mode = render_mode

        self.timestep = 0
        self.limited_timestep = limited_timestep
   
    def render(self):
        return self.engine.render(self.render_mode)

    def step(self, action):
        if isinstance(action, (tuple, list)):
            p1_action, p2_action = action
        else:
            p1_action, p2_action = action, None
            
        if p1_action is None:
            p1_action = self.engine.let_computer_decide_user_input(player_id=0)
        else:
            p1_action = convert_to_user_input(p1_action, 0)
        
        if p2_action is None:
            p2_action = self.engine.let_computer_decide_user_input(player_id=1)
        else:
            p2_action = convert_to_user_input(p2_action, 1)
        
        converted_action = (p1_action, p2_action)
            
        self.timestep += 1
        is_ball_touching_ground = self.engine.step(converted_action)
        self.engine.viewer.update()
        obs = self.engine.get_obs(self.pixel_mode)
        other_obs = self.engine.get_other_obs(self.pixel_mode)
        info = {
            'other_obs': other_obs
        }
        if is_ball_touching_ground or self.timestep >= self.limited_timestep:
            reward = -1 if self.timestep >= self.limited_timestep or self.engine.ball.punch_effect_x < GROUND_HALF_WIDTH else 1
            self.is_player2_serve = (reward == -1)
            return obs, reward, True, True, info
        return obs, 0.0, False, False, info

    def reset(self, options=None, seed: Optional[int]=None, return_info: bool=False) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
        if seed is not None: self.engine.seed(seed)
        if options is None:
            self.engine.reset(self.is_player2_serve)
        elif 'is_player2_serve' in options:
            self.engine.reset(options['is_player2_serve'])
        else:
            raise KeyError
        obs = self.engine.get_obs(self.pixel_mode)
        other_obs = self.engine.get_other_obs(self.pixel_mode)
        info = {
            'other_obs': other_obs
        }
        self.timestep = 0
        return (obs, info)

    def close(self) -> None:
        self.engine.close()

class PikachuVolleyballPixelEnv(PikachuVolleyballEnv):
    pixel_mode = True

class PikachuVolleyballRandomEnv(PikachuVolleyballEnv):   
    more_random = True 
