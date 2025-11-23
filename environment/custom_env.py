import gymnasium as gym
from gymnasium import spaces
import numpy as np

class FireRescueEnv(gym.Env):
    """
    Fixed Fire-Rescue Environment
    Key fixes:
    1. Success detection happens immediately after drop-off
    2. Episode terminates right after success
    3. Clearer reward structure
    4. Distance shaping disabled after drop-off
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, grid_size=10, max_time=200):  # Increased time
        super().__init__()

        self.grid_size = grid_size
        self.max_time = max_time

        # 6 actions: 4 movement + scan + pickup/drop
        self.action_space = spaces.Discrete(6)

        # Observation: [agent_x, agent_y, survivor_x, survivor_y, carrying_flag,
        #               dist_to_survivor, dist_to_door, time_left]
        self.obs_dim = 8
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.obs_dim,), dtype=np.float32
        )

        # Static walls
        self.walls = set([
            (3, 3), (3, 4), (3, 5),
            (6, 2), (6, 3), (6, 4)
        ])

        self.door = np.array([0, 0], dtype=np.int32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Agent starts at door
        self.agent = self.door.copy()

        # Spawn survivor at random free cell
        self.survivor = self._random_free_cell()

        self.carrying = 0
        self.time_left = self.max_time
        
        # Reset distance tracking
        self.last_s_dist = None
        self.last_d_dist = None

        return self._get_obs(), {}

    def _random_free_cell(self):
        """Pick any free location not in a wall and not at the door."""
        while True:
            x = np.random.randint(0, self.grid_size)
            y = np.random.randint(0, self.grid_size)
            if (x, y) not in self.walls and not np.array_equal([x, y], self.door):
                return np.array([x, y], dtype=np.int32)

    def step(self, action):
        reward = -0.01   # Small step penalty to encourage efficiency
        done = False
        truncated = False
        info = {}

        # Decrease time
        self.time_left -= 1
        
        # Check timeout FIRST
        if self.time_left <= 0:
            info["timeout"] = True
            info["success"] = False
            return self._get_obs(), -10, False, True, info  # truncated=True

        # === EXECUTE ACTION ===================================

        if action in [0, 1, 2, 3]:  # Movement
            move_reward = self._move(action)
            reward += move_reward

        elif action == 4:  # Scan
            manhattan_dist = np.linalg.norm(self.agent - self.survivor, ord=1)
            if manhattan_dist <= 1:
                reward += 2  # Survivor nearby
            else:
                reward -= 0.5  # Wrong scan

        elif action == 5:  # Pickup / Drop
            pickup_drop_reward = self._pickup_or_drop()
            reward += pickup_drop_reward
            
            # FIX: Check success immediately after drop-off
            if pickup_drop_reward == 20:  # Just completed drop-off
                # Verify survivor is at door
                if np.array_equal(self.survivor, self.door):
                    reward += 30  # Success bonus
                    done = True
                    info["success"] = True
                    return self._get_obs(), reward, done, truncated, info

        # === DISTANCE SHAPING (only if not done) ==============
        
        if not done:
            if self.carrying == 0:
                # Not carrying - reward getting closer to survivor
                curr = self._dist(self.agent, self.survivor)
                if self.last_s_dist is not None and curr < self.last_s_dist:
                    reward += 0.5  # Reduced from 1
                self.last_s_dist = curr

            else:  # carrying == 1
                # Carrying - reward getting closer to door
                curr = self._dist(self.agent, self.door)
                if self.last_d_dist is not None and curr < self.last_d_dist:
                    reward += 0.5  # Reduced from 1
                self.last_d_dist = curr

        return self._get_obs(), reward, done, truncated, info

    def _move(self, action):
        """Execute movement action. Returns reward."""
        x, y = self.agent.copy()
        
        # Calculate new position
        if action == 0:   # Up (North - decrease y)
            y -= 1
        elif action == 1: # Down (South - increase y)
            y += 1
        elif action == 2: # Left (West - decrease x)
            x -= 1
        elif action == 3: # Right (East - increase x)
            x += 1

        # Check boundaries
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return -1  # Out of bounds penalty

        # Check walls
        if (x, y) in self.walls:
            return -1  # Wall collision penalty

        # Valid move
        self.agent = np.array([x, y], dtype=np.int32)
        return 0  # Neutral reward for valid move

    def _pickup_or_drop(self):
        """Handle pickup and drop-off actions. Returns reward."""
        
        # Attempt pickup
        if self.carrying == 0 and np.array_equal(self.agent, self.survivor):
            self.carrying = 1
            return 10  # Pickup reward

        # Attempt drop-off
        if self.carrying == 1 and np.array_equal(self.agent, self.door):
            self.carrying = 0
            self.survivor = self.door.copy()  # Move survivor to door
            return 20  # Drop-off reward

        # Invalid action
        return -0.5

    def _dist(self, a, b):
        """Manhattan distance between two points."""
        return np.linalg.norm(a - b, ord=1)

    def _get_obs(self):
        """Return normalized observation vector."""
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
        """Simple text rendering."""
        print(f"Agent: {self.agent}, Carrying: {self.carrying}")
        print(f"Survivor: {self.survivor}, Door: {self.door}")
        print(f"Time left: {self.time_left}")

    def close(self):
        pass