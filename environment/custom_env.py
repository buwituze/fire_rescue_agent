import gymnasium as gym
from gymnasium import spaces
import numpy as np

class FireRescueEnv(gym.Env):
    """
    Simplified Fire-Rescue Environment
    Designed specifically so DQN/PPO can learn efficiently.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, grid_size=10, max_time=120):
        super().__init__()

        self.grid_size = grid_size
        self.max_time = max_time

        # 4 movement + scan + pickup/drop
        self.action_space = spaces.Discrete(6)

        # Simple observation (VERY IMPORTANT)
        # [agent_x, agent_y,
        #  survivor_x, survivor_y,
        #  carrying_flag,
        #  dist_to_survivor,
        #  dist_to_door,
        #  time_left]
        self.obs_dim = 8
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.obs_dim,), dtype=np.float32
        )

        # STATIC MAP (just a few walls)
        self.walls = set([
            (3, 3), (3, 4), (3, 5),
            (6, 2), (6, 3), (6, 4)
        ])

        self.door = np.array([0, 0], dtype=np.int32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Agent always starts at door
        self.agent = self.door.copy()

        # One survivor per episode
        self.survivor = self._random_free_cell()

        self.carrying = 0
        self.time_left = self.max_time

        return self._get_obs(), {}

    def _random_free_cell(self):
        """Pick any free location not in a wall and not at the door."""
        while True:
            x = np.random.randint(0, self.grid_size)
            y = np.random.randint(0, self.grid_size)
            if (x, y) not in self.walls and (x, y) != tuple(self.door):
                return np.array([x, y], dtype=np.int32)

    def step(self, action):
        reward = -0.01   # tiny step penalty
        done = False
        info = {}

        self.time_left -= 1
        if self.time_left <= 0:
            return self._get_obs(), -10, True, False, info

        # === ACTIONS =======================================

        if action in [0, 1, 2, 3]:  # movement
            reward += self._move(action)

        elif action == 4:  # scan
            if np.linalg.norm(self.agent - self.survivor, ord=1) <= 1:
                reward += 2
            else:
                reward -= 0.5

        elif action == 5:  # pickup / drop
            reward += self._pickup_or_drop()

        # === DISTANCE SHAPING ===============================

        if self.carrying == 0:
            prev = getattr(self, "last_s_dist", None)
            curr = self._dist(self.agent, self.survivor)
            if prev is not None and curr < prev:
                reward += 1
            self.last_s_dist = curr

        else:
            prev = getattr(self, "last_d_dist", None)
            curr = self._dist(self.agent, self.door)
            if prev is not None and curr < prev:
                reward += 1
            self.last_d_dist = curr

        # === SUCCESS ========================================

        if self.carrying == 0 and np.array_equal(self.survivor, self.door):
            reward += 30
            done = True
            info["success"] = True

        return self._get_obs(), reward, done, False, info

    def _move(self, action):
        x, y = self.agent
        if action == 0: y -= 1
        elif action == 1: y += 1
        elif action == 2: x -= 1
        elif action == 3: x += 1

        # Out of bounds
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return -1

        # Wall hit
        if (x, y) in self.walls:
            return -1

        self.agent = np.array([x, y], dtype=np.int32)
        return 0  # neutral

    def _pickup_or_drop(self):
        # pickup
        if self.carrying == 0 and np.array_equal(self.agent, self.survivor):
            self.carrying = 1
            return 10

        # drop
        if self.carrying == 1 and np.array_equal(self.agent, self.door):
            self.carrying = 0
            self.survivor = self.door.copy()
            return 20

        return -0.5

    def _dist(self, a, b):
        return np.linalg.norm(a - b, ord=1)

    def _get_obs(self):
        return np.array([
            self.agent[0] / self.grid_size,
            self.agent[1] / self.grid_size,
            self.survivor[0] / self.grid_size,
            self.survivor[1] / self.grid_size,
            float(self.carrying),
            self._dist(self.agent, self.survivor) / (self.grid_size * 2),
            self._dist(self.agent, self.door) / (self.grid_size * 2),
            self.time_left / self.max_time
        ], dtype=np.float32)

    def render(self):
        print("Agent:", self.agent, "Carrying:", self.carrying)
        print("Survivor:", self.survivor, "Door:", self.door)

    def close(self):
        pass
