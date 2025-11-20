"""
Fire-Rescue Reinforcement Learning Environment
Custom Gymnasium Environment for Mission-Based RL

Features:
- Grid-based rescue mission
- 1-2 survivors per episode (humans priority, then pets)
- Static fire zones
- Wall obstacles
- Time limit: 180 steps (3 minutes)
- Comprehensive reward structure
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

class FireRescueEnv(gym.Env):
    """
    Fire-Rescue Environment
    
    Agent must navigate a burning building, locate survivors (humans/pets),
    and bring them to the exit door before time runs out.
    
    Action Space: Discrete(6)
        0: Move North
        1: Move South
        2: Move West
        3: Move East
        4: Scan (detect nearby survivors)
        5: Pick Up / Drop Off survivor
    
    Observation Space: Box (continuous vector)
        - Agent position (normalized)
        - Door position (normalized)
        - Time remaining (normalized)
        - Survivor positions and types
        - Survivor alive flags
        - Walls map (flattened)
        - Fire map (flattened)
        - Carrying flag
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, grid_size=10, max_survivors=2, max_time=180):
        super().__init__()

        self.grid_size = grid_size
        self.max_survivors = max_survivors  # Maximum 2 survivors
        self.max_time = max_time  # 180 steps = 3 minutes

        # ACTION SPACE: 6 discrete actions
        # 0=North, 1=South, 2=West, 3=East, 4=Scan, 5=PickUp/DropOff
        self.action_space = spaces.Discrete(6)

        # OBSERVATION SPACE: Calculate dimension
        self.obs_dim = (
            2 +                        # agent position (x, y)
            2 +                        # door position (x, y)
            1 +                        # time remaining (normalized)
            self.max_survivors * 2 +   # survivor positions (x, y for each)
            self.max_survivors +       # survivor alive flags
            self.max_survivors +       # survivor type (1=human, 0=pet)
            self.grid_size ** 2 +      # walls map (flattened grid)
            self.grid_size ** 2 +      # fire map (flattened grid)
            1                          # carrying flag (0 or 1)
        )

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.obs_dim,), dtype=np.float32
        )

        # Initialize static maps
        self.walls = np.zeros((self.grid_size, self.grid_size))
        self.fire = np.zeros((self.grid_size, self.grid_size))
        
        # Episode tracking
        self.episode_reward = 0
        self.survivors_rescued = 0
        self.total_survivors = 0

    # ═══════════════════════════════════════════════════════════
    #   RESET ENVIRONMENT
    # ═══════════════════════════════════════════════════════════
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)

        # Agent starts at door (top-left corner [0, 0])
        self.door_pos = np.array([0, 0], dtype=np.int32)
        self.agent_pos = self.door_pos.copy()

        # Time counter (180 steps = 3 minutes)
        self.time_left = self.max_time

        # Initialize survivor arrays
        self.survivor_positions = np.zeros((self.max_survivors, 2), dtype=np.int32)
        self.survivor_alive = np.zeros(self.max_survivors, dtype=np.float32)
        self.survivor_type = np.zeros(self.max_survivors, dtype=np.float32)  # 1=human, 0=pet

        # SPAWN 1-2 SURVIVORS RANDOMLY
        n_survivors = np.random.randint(1, self.max_survivors + 1)
        self.total_survivors = n_survivors

        # Get free cells (not door)
        free_cells = [(x, y) for x in range(self.grid_size)
                             for y in range(self.grid_size)
                             if (x, y) != tuple(self.door_pos)]

        # Randomly place survivors
        chosen_indices = np.random.choice(len(free_cells), n_survivors, replace=False)

        for i, pos_idx in enumerate(chosen_indices):
            x, y = free_cells[pos_idx]
            self.survivor_positions[i] = np.array([x, y])
            self.survivor_alive[i] = 1.0
            
            # PRIORITIZE HUMANS: First survivor is always human, rest can be pets
            if i == 0:
                self.survivor_type[i] = 1.0  # Human (priority)
            else:
                self.survivor_type[i] = np.random.choice([1.0, 0.0])  # Random human or pet

        # Agent state
        self.carrying = 0  # 0=empty, 1=carrying survivor
        self.carrying_type = 0  # Track what type we're carrying
        self.survivors_rescued = 0
        self.episode_reward = 0

        # Generate static environment (walls and fire)
        self._generate_walls()
        self._generate_fire()
        
        # Distance tracking for approach reward
        self.prev_min_distance = self._get_min_distance_to_survivor()
        
        # Track door loitering
        self.steps_at_door_without_survivor = 0

        obs = self._get_obs()
        return obs, {}

    # ═══════════════════════════════════════════════════════════
    #   STEP FUNCTION - MAIN GAME LOOP
    # ═══════════════════════════════════════════════════════════
    def step(self, action):
        """Execute one time step"""
        reward = -0.01  # Time step penalty (urgency)
        done = False
        info = {}
        
        prev_pos = self.agent_pos.copy()

        # ─────────────────────────────────────
        # TIME MANAGEMENT
        # ─────────────────────────────────────
        self.time_left -= 1
        if self.time_left <= 0:
            done = True
            # Episode ends with survivors remaining = FAILURE
            if np.sum(self.survivor_alive) > 0:
                reward -= 10
            info['timeout'] = True
            self.episode_reward += reward
            return self._get_obs(), reward, done, False, info

        # ─────────────────────────────────────
        # EXECUTE ACTION
        # ─────────────────────────────────────
        if action in [0, 1, 2, 3]:  # MOVEMENT
            move_reward = self._move(action, prev_pos)
            reward += move_reward

        elif action == 4:  # SCAN
            reward += self._scan_reward()

        elif action == 5:  # PICK UP / DROP OFF
            reward += self._pickup_or_drop_reward()

        # ─────────────────────────────────────
        # APPROACH REWARD (getting closer to survivors)
        # ─────────────────────────────────────
        if not self.carrying and np.sum(self.survivor_alive) > 0:
            curr_distance = self._get_min_distance_to_survivor()
            if curr_distance < self.prev_min_distance:
                reward += 0.2  # Approaches survivor (distance decreases)
            self.prev_min_distance = curr_distance

        # ─────────────────────────────────────
        # DOOR LOITERING PENALTY
        # ─────────────────────────────────────
        if np.array_equal(self.agent_pos, self.door_pos) and not self.carrying:
            self.steps_at_door_without_survivor += 1
            if self.steps_at_door_without_survivor > 3:
                reward -= 0.2  # Wastes time near door instead of exiting
        else:
            self.steps_at_door_without_survivor = 0

        # ─────────────────────────────────────
        # SUCCESS CONDITION: All survivors rescued
        # ─────────────────────────────────────
        if np.sum(self.survivor_alive) == 0 and self.survivors_rescued == self.total_survivors:
            done = True
            reward += 15  # All survivors rescued
            info['success'] = True

        self.episode_reward += reward
        return self._get_obs(), reward, done, False, info

    # ═══════════════════════════════════════════════════════════
    #   MOVEMENT LOGIC
    # ═══════════════════════════════════════════════════════════
    def _move(self, action, prev_pos):
        """
        Handle movement actions
        
        Rewards:
        - Valid move: -0.05 (encourages efficiency)
        - Hit wall: -1.0 (penalty for invalid move)
        """
        x, y = self.agent_pos

        # Determine new position based on action
        if action == 0:  # North
            new = (x, y - 1)
        elif action == 1:  # South
            new = (x, y + 1)
        elif action == 2:  # West
            new = (x - 1, y)
        else:  # East (action == 3)
            new = (x + 1, y)

        nx, ny = new
        
        # Check boundaries (out of bounds = wall)
        if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
            return -1.0  # Hitting a wall
        
        # Check wall collision
        if self.walls[ny, nx] == 1:
            return -1.0  # Hitting a wall
        
        # Valid move - update position
        self.agent_pos = np.array([nx, ny])
        return -0.05  # Move to a valid cell (empty or fire)

    # ═══════════════════════════════════════════════════════════
    #   SCAN ACTION
    # ═══════════════════════════════════════════════════════════
    def _scan_reward(self):
        """
        Scan for nearby survivors (within 1 cell Manhattan distance)
        
        Rewards:
        - Survivor nearby: +0.2 (good use of scan)
        - No survivor: -0.1 (discourage spam)
        """
        ax, ay = self.agent_pos
        survivor_nearby = False
        
        for i in range(self.max_survivors):
            if self.survivor_alive[i] == 1:
                sx, sy = self.survivor_positions[i]
                manhattan_dist = abs(ax - sx) + abs(ay - sy)
                if manhattan_dist <= 1:
                    survivor_nearby = True
                    break
        
        if survivor_nearby:
            return 0.2  # Scan (when survivor is nearby)
        else:
            return -0.1  # Scan (no survivor nearby)

    # ═══════════════════════════════════════════════════════════
    #   PICKUP / DROP OFF LOGIC
    # ═══════════════════════════════════════════════════════════
    def _pickup_or_drop_reward(self):
        """
        Handle picking up and dropping off survivors
        
        Rewards:
        - Pick up human: +5.0
        - Pick up pet: +3.0 (lower priority)
        - Return to door with survivor: +10.0
        - Drop survivor at door: +5.0
        Total rescue: +15.0 (or +13.0 for pet)
        """
        ax, ay = self.agent_pos

        # ─────────────────────────────────────
        # DROP OFF at door
        # ─────────────────────────────────────
        if self.carrying == 1 and np.array_equal(self.agent_pos, self.door_pos):
            self.carrying = 0
            self.survivors_rescued += 1
            
            # Bonus based on survivor type
            if self.carrying_type == 1.0:  # Human
                return 10.0 + 5.0  # Returns to door (10) + Drops survivor (5)
            else:  # Pet
                return 10.0 + 3.0  # Slightly lower reward for pet
        
        # ─────────────────────────────────────
        # PICK UP survivor
        # ─────────────────────────────────────
        if self.carrying == 0:
            for i in range(self.max_survivors):
                if self.survivor_alive[i] == 1:
                    if np.array_equal(self.survivor_positions[i], self.agent_pos):
                        self.survivor_alive[i] = 0
                        self.carrying = 1
                        self.carrying_type = self.survivor_type[i]
                        
                        # Reward prioritizes humans
                        if self.survivor_type[i] == 1.0:  # Human
                            return 5.0  # Picks up a survivor (human)
                        else:  # Pet
                            return 3.0  # Picks up a survivor (pet - lower priority)
        
        # Invalid action (nothing to pickup/drop)
        return -0.1

    # ═══════════════════════════════════════════════════════════
    #   HELPER FUNCTIONS
    # ═══════════════════════════════════════════════════════════
    def _get_min_distance_to_survivor(self):
        """Calculate minimum Manhattan distance to any alive survivor"""
        if np.sum(self.survivor_alive) == 0:
            return float('inf')
        
        ax, ay = self.agent_pos
        min_dist = float('inf')
        
        for i in range(self.max_survivors):
            if self.survivor_alive[i] == 1:
                sx, sy = self.survivor_positions[i]
                dist = abs(ax - sx) + abs(ay - sy)
                min_dist = min(min_dist, dist)
        
        return min_dist

    def _get_obs(self):
        """Build observation vector"""
        obs = []

        # Agent and door positions (normalized)
        obs.extend(self.agent_pos / self.grid_size)
        obs.extend(self.door_pos / self.grid_size)
        
        # Time remaining (normalized)
        obs.append(self.time_left / self.max_time)

        # Survivor data
        for i in range(self.max_survivors):
            obs.extend(self.survivor_positions[i] / self.grid_size)
        obs.extend(self.survivor_alive)
        obs.extend(self.survivor_type)

        # Environment maps (flattened)
        obs.extend(self.walls.flatten())
        obs.extend(self.fire.flatten())

        # Carrying flag
        obs.append(self.carrying)

        return np.array(obs, dtype=np.float32)

    # ═══════════════════════════════════════════════════════════
    #   ENVIRONMENT GENERATION
    # ═══════════════════════════════════════════════════════════
    def _generate_walls(self):
        """Generate random walls (10% density)"""
        self.walls = (np.random.rand(self.grid_size, self.grid_size) < 0.1).astype(float)
        self.walls[0, 0] = 0  # Door always clear
        
        # Ensure survivors don't spawn in walls
        for i in range(self.max_survivors):
            if self.survivor_alive[i] == 1:
                sx, sy = self.survivor_positions[i]
                self.walls[sy, sx] = 0

    def _generate_fire(self):
        """Generate random fire patches (15% density) - STATIC"""
        self.fire = (np.random.rand(self.grid_size, self.grid_size) < 0.15).astype(float)
        self.fire[0, 0] = 0  # Door has no fire

    # ═══════════════════════════════════════════════════════════
    #   RENDER (Console output for now, Unity later)
    # ═══════════════════════════════════════════════════════════
    def render(self):
        """
        Simple text-based rendering for debugging
        Unity visualization will be implemented after training
        """
        if self.metadata["render_modes"][0] == "human":
            print(f"\n{'='*50}")
            print(f"Step: {self.max_time - self.time_left}/{self.max_time}")
            print(f"Agent Position: {self.agent_pos}")
            print(f"Carrying: {'Yes' if self.carrying else 'No'}")
            print(f"Survivors Alive: {int(np.sum(self.survivor_alive))}/{self.total_survivors}")
            print(f"Survivors Rescued: {self.survivors_rescued}/{self.total_survivors}")
            print(f"Episode Reward: {self.episode_reward:.2f}")
            print(f"{'='*50}")

    def close(self):
        """Cleanup"""
        pass


# ═══════════════════════════════════════════════════════════
#   REGISTRATION (for Gym compatibility)
# ═══════════════════════════════════════════════════════════
from gymnasium.envs.registration import register

register(
    id='FireRescue-v0',
    entry_point='custom_env:FireRescueEnv',
    max_episode_steps=180,
)