import pygame
import numpy as np
from typing import Optional, Tuple

class FireRescueRenderer:
    """
    Advanced Pygame-based visualization for Fire-Rescue Environment
    Features:
    - High-quality 2D rendering with visual effects
    - Real-time agent state display
    - Animated transitions
    - Information panel with metrics
    """
    
    def __init__(self, grid_size: int = 10, cell_size: int = 60):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.width = grid_size * cell_size + 250  # Extra space for info panel
        self.height = grid_size * cell_size + 100  # Extra space for header
        
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Fire-Rescue Mission - RL Visualization")
        self.clock = pygame.time.Clock()
        
        # Colors (professional palette)
        self.COLORS = {
            'background': (240, 240, 245),
            'grid_line': (200, 200, 210),
            'wall': (60, 60, 70),
            'wall_highlight': (80, 80, 90),
            'agent': (30, 144, 255),  # Dodger blue
            'agent_carrying': (255, 140, 0),  # Dark orange
            'survivor': (220, 20, 60),  # Crimson
            'survivor_rescued': (50, 205, 50),  # Lime green
            'door': (34, 139, 34),  # Forest green
            'door_glow': (144, 238, 144),  # Light green
            'text': (40, 40, 50),
            'text_highlight': (255, 69, 0),  # Red-orange
            'panel_bg': (255, 255, 255),
            'fire': (255, 69, 0),  # Fire effect
            'smoke': (128, 128, 128),  # Smoke effect
        }
        
        # Fonts
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 22)
        
        # Offset for grid rendering
        self.grid_offset_x = 50
        self.grid_offset_y = 80
        
        # Animation state
        self.agent_pulse = 0
        self.fire_animation = 0
        
    def render(self, env, step_count: int = 0, total_reward: float = 0.0, 
               episode: int = 0, info: Optional[dict] = None):
        """
        Render the current state of the environment
        
        Args:
            env: FireRescueEnv instance
            step_count: Current step number
            total_reward: Cumulative reward
            episode: Episode number
            info: Additional info dict from environment
        """
        # Clear screen
        self.screen.fill(self.COLORS['background'])
        
        # Draw header
        self._draw_header(episode, step_count, total_reward, env.time_left)
        
        # Draw grid and walls
        self._draw_grid()
        self._draw_walls(env.walls)
        
        # Draw door with glow effect
        self._draw_door(env.door)
        
        # Draw survivor
        self._draw_survivor(env.survivor, env.carrying)
        
        # Draw agent
        self._draw_agent(env.agent, env.carrying)
        
        # Draw info panel
        self._draw_info_panel(env, step_count, total_reward, info)
        
        # Update animation states
        self._update_animations()
        
        # Update display
        pygame.display.flip()
        self.clock.tick(10)  # 10 FPS for visibility
        
    def _draw_header(self, episode: int, step: int, reward: float, time_left: int):
        """Draw header with episode info"""
        header_text = f"Fire-Rescue Mission | Episode: {episode} | Step: {step}"
        text_surface = self.font_large.render(header_text, True, self.COLORS['text'])
        self.screen.blit(text_surface, (20, 20))
        
        # Time indicator with color coding
        time_color = self.COLORS['text']
        if time_left < 50:
            time_color = self.COLORS['text_highlight']
        
        time_text = f"Time: {time_left}"
        time_surface = self.font_medium.render(time_text, True, time_color)
        self.screen.blit(time_surface, (self.width - 250, 25))
        
    def _draw_grid(self):
        """Draw grid lines"""
        for i in range(self.grid_size + 1):
            # Vertical lines
            x = self.grid_offset_x + i * self.cell_size
            pygame.draw.line(
                self.screen, 
                self.COLORS['grid_line'],
                (x, self.grid_offset_y),
                (x, self.grid_offset_y + self.grid_size * self.cell_size),
                1
            )
            # Horizontal lines
            y = self.grid_offset_y + i * self.cell_size
            pygame.draw.line(
                self.screen,
                self.COLORS['grid_line'],
                (self.grid_offset_x, y),
                (self.grid_offset_x + self.grid_size * self.cell_size, y),
                1
            )
    
    def _draw_walls(self, walls: set):
        """Draw walls with 3D effect"""
        for wall in walls:
            x, y = wall
            cell_x = self.grid_offset_x + x * self.cell_size
            cell_y = self.grid_offset_y + y * self.cell_size
            
            # Main wall
            pygame.draw.rect(
                self.screen,
                self.COLORS['wall'],
                (cell_x + 2, cell_y + 2, self.cell_size - 4, self.cell_size - 4)
            )
            
            # Highlight for 3D effect
            pygame.draw.rect(
                self.screen,
                self.COLORS['wall_highlight'],
                (cell_x + 2, cell_y + 2, self.cell_size - 4, 8)
            )
            
            # Wall texture (brick pattern)
            for i in range(0, self.cell_size - 4, 15):
                for j in range(0, self.cell_size - 4, 8):
                    pygame.draw.rect(
                        self.screen,
                        self.COLORS['wall_highlight'],
                        (cell_x + 4 + i, cell_y + 4 + j, 12, 6),
                        1
                    )
    
    def _draw_door(self, door: np.ndarray):
        """Draw door/exit with glow effect"""
        x, y = door
        cell_x = self.grid_offset_x + x * self.cell_size
        cell_y = self.grid_offset_y + y * self.cell_size
        center_x = cell_x + self.cell_size // 2
        center_y = cell_y + self.cell_size // 2
        
        # Glow effect (multiple circles with decreasing alpha)
        for radius in range(30, 15, -3):
            alpha = 50 - (30 - radius) * 2
            glow_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(
                glow_surface,
                (*self.COLORS['door_glow'], alpha),
                (radius, radius),
                radius
            )
            self.screen.blit(glow_surface, (center_x - radius, center_y - radius))
        
        # Main door
        pygame.draw.rect(
            self.screen,
            self.COLORS['door'],
            (cell_x + 8, cell_y + 8, self.cell_size - 16, self.cell_size - 16),
            0,
            border_radius=5
        )
        
        # Door symbol (arrow out)
        pygame.draw.polygon(
            self.screen,
            self.COLORS['door_glow'],
            [
                (center_x, center_y - 12),
                (center_x - 8, center_y),
                (center_x + 8, center_y),
                (center_x, center_y + 12)
            ]
        )
        
        # Label
        label = self.font_small.render("EXIT", True, (255, 255, 255))
        label_rect = label.get_rect(center=(center_x, center_y + 20))
        self.screen.blit(label, label_rect)
    
    def _draw_survivor(self, survivor: np.ndarray, carrying: int):
        """Draw survivor with animation"""
        if carrying == 1:
            return  # Don't draw if being carried
        
        x, y = survivor
        cell_x = self.grid_offset_x + x * self.cell_size
        cell_y = self.grid_offset_y + y * self.cell_size
        center_x = cell_x + self.cell_size // 2
        center_y = cell_y + self.cell_size // 2
        
        # Pulsing danger indicator
        pulse = abs(np.sin(self.fire_animation / 10)) * 10
        
        # Danger zone
        pygame.draw.circle(
            self.screen,
            self.COLORS['fire'],
            (center_x, center_y),
            20 + int(pulse),
            2
        )
        
        # Survivor body
        pygame.draw.circle(
            self.screen,
            self.COLORS['survivor'],
            (center_x, center_y),
            15
        )
        
        # SOS text
        sos_text = self.font_small.render("SOS", True, (255, 255, 255))
        sos_rect = sos_text.get_rect(center=(center_x, center_y))
        self.screen.blit(sos_text, sos_rect)
    
    def _draw_agent(self, agent: np.ndarray, carrying: int):
        """Draw agent (rescue robot/person)"""
        x, y = agent
        cell_x = self.grid_offset_x + x * self.cell_size
        cell_y = self.grid_offset_y + y * self.cell_size
        center_x = cell_x + self.cell_size // 2
        center_y = cell_y + self.cell_size // 2
        
        # Color based on carrying state
        agent_color = self.COLORS['agent_carrying'] if carrying else self.COLORS['agent']
        
        # Pulsing effect
        pulse = abs(np.sin(self.agent_pulse / 15)) * 5
        
        # Agent shadow
        shadow_surface = pygame.Surface((40, 40), pygame.SRCALPHA)
        pygame.draw.circle(shadow_surface, (0, 0, 0, 50), (20, 20), 18)
        self.screen.blit(shadow_surface, (center_x - 20 + 2, center_y - 20 + 2))
        
        # Agent body (larger circle)
        pygame.draw.circle(
            self.screen,
            agent_color,
            (center_x, center_y),
            18 + int(pulse)
        )
        
        # Agent highlight
        pygame.draw.circle(
            self.screen,
            (255, 255, 255),
            (center_x - 5, center_y - 5),
            5
        )
        
        # If carrying, show survivor indicator
        if carrying:
            survivor_icon = self.font_small.render("👤", True, (255, 255, 255))
            icon_rect = survivor_icon.get_rect(center=(center_x, center_y))
            self.screen.blit(survivor_icon, icon_rect)
            
            # "CARRYING" label
            label = self.font_small.render("CARRYING", True, self.COLORS['text_highlight'])
            label_rect = label.get_rect(center=(center_x, center_y + 28))
            self.screen.blit(label, label_rect)
    
    def _draw_info_panel(self, env, step: int, reward: float, info: Optional[dict]):
        """Draw information panel with metrics"""
        panel_x = self.grid_offset_x + self.grid_size * self.cell_size + 20
        panel_y = self.grid_offset_y
        panel_width = 220
        panel_height = 400
        
        # Panel background
        pygame.draw.rect(
            self.screen,
            self.COLORS['panel_bg'],
            (panel_x, panel_y, panel_width, panel_height),
            border_radius=10
        )
        pygame.draw.rect(
            self.screen,
            self.COLORS['grid_line'],
            (panel_x, panel_y, panel_width, panel_height),
            2,
            border_radius=10
        )
        
        # Title
        title = self.font_medium.render("Mission Status", True, self.COLORS['text'])
        self.screen.blit(title, (panel_x + 10, panel_y + 10))
        
        # Metrics
        y_offset = panel_y + 50
        metrics = [
            ("Step:", f"{step}"),
            ("Reward:", f"{reward:.2f}"),
            ("Carrying:", "YES" if env.carrying else "NO"),
            ("Time Left:", f"{env.time_left}"),
        ]
        
        # Add info dict metrics if available
        if info:
            if "wall_collisions" in info:
                metrics.append(("Collisions:", f"{info['wall_collisions']}"))
            if "scan_attempts" in info:
                metrics.append(("Scans:", f"{info['scan_attempts']}"))
        
        for label, value in metrics:
            label_surface = self.font_small.render(label, True, self.COLORS['text'])
            value_surface = self.font_small.render(str(value), True, self.COLORS['text_highlight'])
            self.screen.blit(label_surface, (panel_x + 10, y_offset))
            self.screen.blit(value_surface, (panel_x + 130, y_offset))
            y_offset += 30
        
        # Legend
        legend_y = panel_y + panel_height - 150
        legend_title = self.font_small.render("Legend:", True, self.COLORS['text'])
        self.screen.blit(legend_title, (panel_x + 10, legend_y))
        
        legend_items = [
            (self.COLORS['agent'], "Agent (Empty)"),
            (self.COLORS['agent_carrying'], "Agent (Carrying)"),
            (self.COLORS['survivor'], "Survivor"),
            (self.COLORS['door'], "Exit Door"),
            (self.COLORS['wall'], "Wall"),
        ]
        
        legend_y += 30
        for color, text in legend_items:
            pygame.draw.circle(self.screen, color, (panel_x + 20, legend_y), 8)
            text_surface = self.font_small.render(text, True, self.COLORS['text'])
            self.screen.blit(text_surface, (panel_x + 35, legend_y - 8))
            legend_y += 22
    
    def _update_animations(self):
        """Update animation counters"""
        self.agent_pulse += 1
        self.fire_animation += 1
        if self.agent_pulse > 1000:
            self.agent_pulse = 0
        if self.fire_animation > 1000:
            self.fire_animation = 0
    
    def handle_events(self) -> bool:
        """Handle pygame events. Returns False if user wants to quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
        return True
    
    def close(self):
        """Clean up pygame resources"""
        pygame.quit()