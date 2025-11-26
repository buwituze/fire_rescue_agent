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

    def __init__(self, grid_size=10, max_time=250):  # Increased time for training
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
        self.survivor_initial_pos = self.survivor.copy()  # Track initial position

        self.carrying = 0
        self.time_left = self.max_time
        
        # Reset distance tracking
        self.last_s_dist = None
        self.last_d_dist = None
        
        # Reset episode metrics
        self.wall_collisions = 0
        self.total_moves = 0
        self.scan_attempts = 0
        self.useful_scans = 0
        self.pickup_attempts = 0
        self.survivor_found_step = None  # Track when survivor was first reached

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
        info = {"success": False}  # Initialize success flag to False

        # Decrease time
        self.time_left -= 1
        current_step = self.max_time - self.time_left
        
        # Check timeout FIRST
        if self.time_left <= 0:
            info["timeout"] = True
            info["success"] = False
            # Add metrics to info
            self._add_metrics_to_info(info, current_step)
            return self._get_obs(), -10, False, True, info  # truncated=True

        # === EXECUTE ACTION ===================================

        if action in [0, 1, 2, 3]:  # Movement
            self.total_moves += 1
            move_reward = self._move(action)
            reward += move_reward
            if move_reward == -1:  # Wall collision or out of bounds
                self.wall_collisions += 1

        elif action == 4:  # Scan
            self.scan_attempts += 1
            # Scan gives bonus when near survivor to encourage discovery
            # BUT only when NOT carrying (prevent exploitation when carrying survivor)
            if self.carrying == 0:
                manhattan_dist = np.linalg.norm(self.agent - self.survivor, ord=1)
                if manhattan_dist <= 1.5:  # Slightly larger radius
                    self.useful_scans += 1
                    reward += 0.5  # Moderate reward for successful scan
                else:
                    reward -= 0.05  # Small penalty for wrong scan
            else:
                # If carrying, scanning is useless - STRONG penalty
                reward -= 1.0  # Strong penalty for scanning while carrying (from -0.2)

        elif action == 5:  # Pickup / Drop
            self.pickup_attempts += 1
            pickup_drop_reward = self._pickup_or_drop()
            reward += pickup_drop_reward
            
            # Check success immediately after drop-off
            if pickup_drop_reward == 30:  # Just completed drop-off (updated from 20)
                # Verify survivor is at door
                if np.array_equal(self.survivor, self.door):
                    reward += 50  # Success bonus (total +80 for complete drop-off)
                    done = True
                    info["success"] = True
                    self._add_metrics_to_info(info, current_step)
                    return self._get_obs(), reward, done, truncated, info
        
        # Track when survivor is first reached
        if self.survivor_found_step is None and np.array_equal(self.agent, self.survivor_initial_pos):
            self.survivor_found_step = current_step

        # === DISTANCE SHAPING - Two-Phase Guidance ====================
        # Phase 1: Guide agent TO survivor (when not carrying)
        # Phase 2: Guide agent TO door (when carrying survivor)
        
        if action in [0, 1, 2, 3]:  # Only apply to movement actions
            if self.carrying == 0:
                # PHASE 1: Not carrying - guide toward survivor
                current_s_dist = self._dist(self.agent, self.survivor)
                if self.last_s_dist is None:
                    self.last_s_dist = current_s_dist
                
                if current_s_dist < self.last_s_dist:
                    reward += 0.2  # Moderate reward for approaching survivor
                elif current_s_dist > self.last_s_dist:
                    reward -= 0.1  # Small penalty for moving away
                
                self.last_s_dist = current_s_dist
            
            else:  # self.carrying == 1
                # PHASE 2: Carrying survivor - guide toward door
                current_door_dist = self._dist(self.agent, self.door)
                if self.last_d_dist is None:
                    self.last_d_dist = current_door_dist
                
                if current_door_dist < self.last_d_dist:
                    reward += 0.8  # Strong reward for returning to door with survivor
                elif current_door_dist > self.last_d_dist:
                    reward -= 0.4  # Penalty for moving away from door
                
                self.last_d_dist = current_door_dist

        # Add metrics to info before returning
        self._add_metrics_to_info(info, current_step)
        return self._get_obs(), reward, done, truncated, info
    
    def _add_metrics_to_info(self, info, current_step):
        """Add episode metrics to info dict."""
        info["wall_collisions"] = self.wall_collisions
        info["total_moves"] = self.total_moves
        info["wall_collision_rate"] = self.wall_collisions / max(1, self.total_moves)
        info["scan_attempts"] = self.scan_attempts
        info["useful_scans"] = self.useful_scans
        info["scan_efficiency"] = self.useful_scans / max(1, self.scan_attempts)
        info["pickup_attempts"] = self.pickup_attempts
        info["time_to_find_survivor"] = self.survivor_found_step if self.survivor_found_step else None
        info["episode_length"] = current_step

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
        
        # If carrying survivor, move survivor with agent
        if self.carrying == 1:
            self.survivor = self.agent.copy()
        
        return 0  # Neutral reward for valid move

    def _pickup_or_drop(self):
        """Handle pickup and drop-off actions. Returns reward."""
        
        # Attempt pickup
        if self.carrying == 0 and np.array_equal(self.agent, self.survivor):
            self.carrying = 1
            return 15  # Pickup reward (increased from 10)

        # Attempt drop-off
        if self.carrying == 1 and np.array_equal(self.agent, self.door):
            self.carrying = 0
            self.survivor = self.door.copy()  # Move survivor to door
            return 30  # Drop-off reward (increased from 20)

        # Invalid action
        return -1.0  # Stronger penalty for invalid pickup/drop (from -0.5)

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