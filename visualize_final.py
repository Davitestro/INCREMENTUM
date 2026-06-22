# visualize_strategic.py
import pygame
import torch
import numpy as np
import sys
import os
import glob
import math

sys.path.append('.')

from config.base_config import BaseConfig
from environment.battle_env import BattleEnvironment
from models.attacker import AttackerModel
from models.defender import DefenderModel

class StrategicVisualizer:
    def __init__(self, model_path=None):
        self.config = BaseConfig()
        self.config.map_size = 64
        self.config.max_units = 15
        self.config.max_steps = 300
        
        pygame.init()
        self.screen = pygame.display.set_mode((1200, 800))
        pygame.display.set_caption("Strategic RL - Стратегическая визуализация")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.big_font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 16)
        
        self.env = BattleEnvironment(self.config)
        
        self.attacker = None
        self.defender = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print("⚠️ No model found - running random")
        
        self.paused = False
        self.speed = 2
        self.step_mode = False
        self.step_requested = False
        self.episode_count = 0
        self.running = True
        
        self.strategy_colors = {
            'attacker_advance': (0, 255, 100),
            'attacker_flank': (0, 200, 255),
            'defender_defense': (255, 50, 50),
            'defender_counter': (255, 200, 0),
            'objective': (255, 255, 255),
            'supply': (200, 150, 100)
        }
        
    def load_model(self, path):
        try:
            print(f"📦 Loading model from: {path}")
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            
            self.attacker = AttackerModel(self.config)
            self.defender = DefenderModel(self.config)
            
            self.attacker.load_state_dict(checkpoint['attacker'])
            self.defender.load_state_dict(checkpoint['defender'])
            
            self.attacker.eval()
            self.defender.eval()
            
            print(f"✅ Model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def prepare_input(self, state):
        terrain = state.get('terrain')
        if terrain is not None:
            if isinstance(terrain, np.ndarray):
                if len(terrain.shape) == 2:
                    terrain = torch.FloatTensor(terrain).unsqueeze(0).unsqueeze(0)
                else:
                    terrain = torch.FloatTensor(terrain).unsqueeze(0)
            else:
                terrain = torch.zeros(1, 1, self.config.map_size, self.config.map_size)
        else:
            terrain = torch.zeros(1, 1, self.config.map_size, self.config.map_size)
        
        return {'terrain': terrain, 'units': torch.zeros(1, self.config.max_units, 10)}
    
    def strategy_to_actions(self, strategy, units, is_attacker=True):
        actions = {}
        
        if not strategy:
            return actions
        
        alive_units = [u for u in units if u.alive]
        if not alive_units:
            return actions
        
        if is_attacker:
            lines = strategy.get('advance_lines', [])
        else:
            lines = strategy.get('defense_lines', [])
        
        has_lines = False
        if lines is not None:
            if isinstance(lines, np.ndarray):
                if lines.size > 0:
                    has_lines = True
                    if len(lines.shape) == 3:
                        lines = lines[0]
            elif isinstance(lines, (list, tuple)) and len(lines) > 0:
                has_lines = True
                if isinstance(lines[0], (list, tuple, np.ndarray)):
                    lines = lines[0]
        
        if has_lines and len(lines) > 1:
            for i, unit in enumerate(alive_units):
                idx = i % len(lines)
                target = lines[idx]
                
                if isinstance(target, (list, tuple, np.ndarray)) and len(target) >= 2:
                    target_x = float(target[0])
                    target_y = float(target[1])
                    
                    offset_x = np.random.uniform(-3, 3)
                    offset_y = np.random.uniform(-3, 3)
                    
                    actions[str(unit.id)] = {
                        'move': (target_x + offset_x, target_y + offset_y)
                    }
        
        if not actions:
            if is_attacker:
                enemies = [u for u in self.env.defender_units if u.alive]
            else:
                enemies = [u for u in self.env.attacker_units if u.alive]
            
            if enemies:
                target_x = np.mean([u.x for u in enemies])
                target_y = np.mean([u.y for u in enemies])
            else:
                target_x = self.config.map_size / 2
                target_y = self.config.map_size / 2
            
            for unit in alive_units:
                offset_x = np.random.uniform(-5, 5)
                offset_y = np.random.uniform(-5, 5)
                actions[str(unit.id)] = {
                    'move': (target_x + offset_x, target_y + offset_y)
                }
        
        for unit in alive_units:
            if is_attacker:
                enemies = [u for u in self.env.defender_units if u.alive]
            else:
                enemies = [u for u in self.env.attacker_units if u.alive]
            
            if enemies:
                nearest = min(enemies, key=lambda e: np.sqrt((e.x - unit.x)**2 + (e.y - unit.y)**2))
                dist = np.sqrt((nearest.x - unit.x)**2 + (nearest.y - unit.y)**2)
                if dist < unit.range:
                    if str(unit.id) in actions:
                        actions[str(unit.id)]['attack'] = nearest.id
                    else:
                        actions[str(unit.id)] = {'attack': nearest.id}
        
        return actions
    
    def draw_strategy_lines(self, lines, color, cell_size, label=""):
        if lines is None:
            return
        
        if isinstance(lines, np.ndarray):
            if lines.size == 0:
                return
            if len(lines.shape) == 3:
                lines = lines[0]
        
        points = []
        for p in lines:
            if isinstance(p, (list, tuple, np.ndarray)) and len(p) >= 2:
                x = int(p[0] * cell_size)
                y = int(p[1] * cell_size)
                points.append((x, y))
        
        if len(points) < 2:
            return
        
        if len(points) > 1:
            pygame.draw.lines(self.screen, color, False, points, 4)
        
        for i, p in enumerate(points):
            pygame.draw.circle(self.screen, color, p, 6)
            for r in range(10, 20, 3):
                pygame.draw.circle(self.screen, color, p, r, 1)
            
            if i % 3 == 0:
                num_text = self.small_font.render(str(i+1), True, (255, 255, 255))
                self.screen.blit(num_text, (p[0] + 8, p[1] - 8))
        
        for i in range(0, len(points) - 5, 5):
            p1 = points[i]
            p2 = points[i + 5]
            self.draw_arrow(p1, p2, color)
        
        if label and len(points) > 0:
            mid_idx = len(points) // 2
            mid_point = points[mid_idx]
            text = self.font.render(label, True, color)
            self.screen.blit(text, (mid_point[0] - 30, mid_point[1] - 30))
    
    def draw_arrow(self, start, end, color):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.sqrt(dx**2 + dy**2)
        
        if length > 20:
            angle = math.atan2(dy, dx)
            arrow_len = 15
            arrow_angle = 0.4
            
            tip = end
            left = (end[0] - arrow_len * math.cos(angle - arrow_angle),
                    end[1] - arrow_len * math.sin(angle - arrow_angle))
            right = (end[0] - arrow_len * math.cos(angle + arrow_angle),
                     end[1] - arrow_len * math.sin(angle + arrow_angle))
            
            pygame.draw.polygon(self.screen, color, [tip, left, right])
            
            glow_color = (*color[:3], 30)
            pygame.draw.polygon(self.screen, glow_color, 
                              [(tip[0]-3, tip[1]-3), left, right], 2)
    
    def draw_objectives(self, state, cell_size):
        objectives = state.get('objectives', [])
        
        for obj in objectives:
            x = int(obj.get('x', 0) * cell_size)
            y = int(obj.get('y', 0) * cell_size)
            radius = obj.get('radius', 8)
            value = obj.get('value', 0)
            obj_type = obj.get('type', 'strategic')
            
            if obj_type == 'capture':
                color = (255, 215, 0)
                label = "⭐ CAPTURE"
            else:
                color = (100, 200, 255)
                label = "🎯 STRATEGIC"
            
            pulse = int(5 * math.sin(pygame.time.get_ticks() / 500) + 8)
            for r in range(radius - 3, radius + pulse + 3, 4):
                pygame.draw.circle(self.screen, color, (x, y), r, 2)
            
            pygame.draw.circle(self.screen, color, (x, y), radius)
            pygame.draw.circle(self.screen, (255, 255, 255), (x, y), radius - 2, 2)
            
            if obj_type == 'capture':
                text = self.big_font.render("⭐", True, (255, 215, 0))
                self.screen.blit(text, (x - 15, y - 35))
            
            val_text = self.small_font.render(f"+{value}", True, (255, 255, 255))
            self.screen.blit(val_text, (x - 15, y + radius + 10))
    
    def draw_front_line(self, state, cell_size):
        front_line = state.get('front_line', [])
        if len(front_line) < 2:
            return
        
        points = []
        for p in front_line:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                x = int(p[0] * cell_size)
                y = int(p[1] * cell_size)
                points.append((x, y))
        
        if len(points) > 1:
            dash_length = 15
            gap_length = 10
            for i in range(len(points) - 1):
                p1 = points[i]
                p2 = points[i + 1]
                
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                dist = math.sqrt(dx**2 + dy**2)
                
                if dist > 0:
                    steps = int(dist / (dash_length + gap_length))
                    for j in range(steps):
                        t1 = j * (dash_length + gap_length) / dist
                        t2 = (j * (dash_length + gap_length) + dash_length) / dist
                        
                        if t2 <= 1:
                            start = (int(p1[0] + dx * t1), int(p1[1] + dy * t1))
                            end = (int(p1[0] + dx * t2), int(p1[1] + dy * t2))
                            pygame.draw.line(self.screen, (255, 200, 100), start, end, 3)
            
            mid_idx = len(points) // 2
            mid = points[mid_idx]
            text = self.font.render("⚔️ FRONT LINE", True, (255, 200, 100))
            self.screen.blit(text, (mid[0] - 50, mid[1] - 30))
    
    def render(self, state, info=None, action_info=None):
        self.screen.fill((15, 15, 25))
        
        map_size = self.config.map_size
        cell_size = 800 // map_size
        
        terrain = state.get('terrain', np.zeros((map_size, map_size)))
        
        for y in range(map_size):
            for x in range(map_size):
                rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
                
                if isinstance(terrain, np.ndarray):
                    height = terrain[y, x] if y < terrain.shape[0] and x < terrain.shape[1] else 0
                else:
                    height = 0
                
                if height < 0.2:
                    color = (20, 40, 60)
                elif height < 0.4:
                    color = (40, 80, 40)
                elif height < 0.7:
                    color = (60, 100, 50)
                else:
                    color = (100, 80, 50)
                
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (30, 30, 40), rect, 1)
        
        self.draw_front_line(state, cell_size)
        self.draw_objectives(state, cell_size)
        
        if action_info:
            if 'attacker_lines' in action_info:
                self.draw_strategy_lines(
                    action_info['attacker_lines'], 
                    self.strategy_colors['attacker_advance'], 
                    cell_size,
                    "⚔️ ADVANCE"
                )
            
            if 'defender_lines' in action_info:
                self.draw_strategy_lines(
                    action_info['defender_lines'], 
                    self.strategy_colors['defender_defense'], 
                    cell_size,
                    "🛡️ DEFENSE"
                )
        
        for unit in self.env.attacker_units:
            if unit.alive:
                x = int(unit.x * cell_size + cell_size // 2)
                y = int(unit.y * cell_size + cell_size // 2)
                radius = cell_size // 2 - 2
                
                pygame.draw.circle(self.screen, (0, 30, 20), (x + 3, y + 3), radius)
                
                for i in range(radius, 0, -3):
                    color = (0, 100 + int(100 * (i / radius)), 50 + int(100 * (i / radius)))
                    pygame.draw.circle(self.screen, color, (x, y), i)
                
                pygame.draw.circle(self.screen, (0, 255, 150), (x, y), radius, 2)
                
                bar_width = radius * 2
                bar_height = 5
                bar_x = x - radius
                bar_y = y + radius + 5
                pygame.draw.rect(self.screen, (30, 30, 30), (bar_x, bar_y, bar_width, bar_height))
                hp_width = int(bar_width * (unit.hp / unit.max_hp))
                if hp_width > 0:
                    pygame.draw.rect(self.screen, (0, 255, 100), (bar_x, bar_y, hp_width, bar_height))
                
                type_text = self.small_font.render(f"{unit.unit_type[:3]}", True, (255, 255, 255))
                self.screen.blit(type_text, (x - 10, y - 6))
        
        for unit in self.env.defender_units:
            if unit.alive:
                x = int(unit.x * cell_size + cell_size // 2)
                y = int(unit.y * cell_size + cell_size // 2)
                radius = cell_size // 2 - 2
                
                pygame.draw.circle(self.screen, (30, 0, 0), (x + 3, y + 3), radius)
                
                for i in range(radius, 0, -3):
                    color = (100 + int(100 * (i / radius)), 20, 20)
                    pygame.draw.circle(self.screen, color, (x, y), i)
                
                pygame.draw.circle(self.screen, (255, 100, 100), (x, y), radius, 2)
                
                bar_width = radius * 2
                bar_height = 5
                bar_x = x - radius
                bar_y = y + radius + 5
                pygame.draw.rect(self.screen, (30, 30, 30), (bar_x, bar_y, bar_width, bar_height))
                hp_width = int(bar_width * (unit.hp / unit.max_hp))
                if hp_width > 0:
                    pygame.draw.rect(self.screen, (255, 50, 50), (bar_x, bar_y, hp_width, bar_height))
                
                type_text = self.small_font.render(f"{unit.unit_type[:3]}", True, (255, 255, 255))
                self.screen.blit(type_text, (x - 10, y - 6))
        
        info_x = 820
        info_y = 20
        line_height = 26
        
        model_status = "✅ AI" if self.attacker else "⚠️ Random"
        
        stats_lines = [
            f"🎯 EPISODE: {self.episode_count}",
            f"⏱ STEP: {self.env.step_count}",
            f"⚡ SPEED: {self.speed}x",
            f"🤖 MODE: {model_status}",
            "",
            f"🟢 ATTACKER: {sum(1 for u in self.env.attacker_units if u.alive)}",
            f"🔴 DEFENDER: {sum(1 for u in self.env.defender_units if u.alive)}",
            "",
            "📊 STRATEGY:",
        ]
        
        for line in stats_lines:
            if line == "":
                info_y += 5
                continue
            
            if "📊 STRATEGY:" in line or "🎯 EPISODE:" in line:
                text = self.big_font.render(line, True, (255, 200, 100))
                self.screen.blit(text, (info_x, info_y))
                info_y += line_height + 5
            else:
                text = self.font.render(line, True, (200, 200, 200))
                self.screen.blit(text, (info_x, info_y))
                info_y += line_height
        
        info_y += 10
        legend_items = [
            ("🟢 Green", "Attack/Advance", self.strategy_colors['attacker_advance']),
            ("🔵 Blue", "Flanking", self.strategy_colors['attacker_flank']),
            ("🔴 Red", "Defense", self.strategy_colors['defender_defense']),
            ("🟡 Yellow", "Counter-attack", self.strategy_colors['defender_counter']),
        ]
        
        for label, desc, color in legend_items:
            pygame.draw.line(self.screen, color, (info_x, info_y + 10), (info_x + 30, info_y + 10), 3)
            text = self.small_font.render(f"{label} - {desc}", True, (200, 200, 200))
            self.screen.blit(text, (info_x + 40, info_y))
            info_y += 22
        
        if info:
            info_y += 10
            reward_text = f"🏆 Att: {info.get('attacker_reward', 0):.0f}  Def: {info.get('defender_reward', 0):.0f}"
            text = self.big_font.render(reward_text, True, (255, 255, 100))
            self.screen.blit(text, (info_x, info_y))
        
        info_y += 50
        controls = [
            "CONTROLS:",
            "SPACE - Pause",
            "S - Step mode",
            "→ - Next step",
            "R - Reset",
            "↑↓ - Speed",
            "ESC - Quit"
        ]
        
        for line in controls:
            color = (255, 200, 100) if "CONTROLS:" in line else (150, 150, 150)
            text = self.small_font.render(line, True, color)
            self.screen.blit(text, (info_x, info_y))
            info_y += 20
        
        if self.step_mode:
            step_text = self.big_font.render("⏯ STEP MODE", True, (255, 200, 50))
            self.screen.blit(step_text, (600, 20))
        
        if self.paused:
            pause_text = self.big_font.render("⏸ PAUSED", True, (255, 200, 50))
            self.screen.blit(pause_text, (600, 60))
        
        for i in range(0, map_size + 1, 10):
            x = i * cell_size
            pygame.draw.line(self.screen, (40, 40, 50), (x, 0), (x, 800), 1)
            pygame.draw.line(self.screen, (40, 40, 50), (0, x), (800, x), 1)
            
            if i % 20 == 0 and i < map_size:
                coord_text = self.small_font.render(str(i), True, (60, 60, 80))
                self.screen.blit(coord_text, (x + 2, 802))
                self.screen.blit(coord_text, (802, x + 2))
        
        pygame.display.flip()
    
    def run(self):
        print("=" * 70)
        print("🎯 STRATEGIC RL - СТРАТЕГИЧЕСКАЯ ВИЗУАЛИЗАЦИЯ")
        print("=" * 70)
        print("📌 Карта увеличена до 64x64")
        print("📌 Показаны стратегические линии и цели")
        print("📌 Юниты следуют стратегическим планам")
        print("")
        print("Controls:")
        print("  SPACE - Pause/Resume")
        print("  S - Step mode (ON/OFF)")
        print("  → - Next step (in step mode)")
        print("  R - Reset episode")
        print("  ↑/↓ - Speed up/down")
        print("  ESC - Quit")
        print("=" * 70)
        
        while self.running:
            state = self.env.reset()
            done = False
            total_reward = {'attacker': 0, 'defender': 0}
            step_count = 0
            
            while not done and self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                        done = True
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self.running = False
                            done = True
                        elif event.key == pygame.K_SPACE:
                            self.paused = not self.paused
                        elif event.key == pygame.K_s:
                            self.step_mode = not self.step_mode
                            print(f"Step mode: {'ON' if self.step_mode else 'OFF'}")
                        elif event.key == pygame.K_RIGHT:
                            if self.step_mode:
                                self.step_requested = True
                        elif event.key == pygame.K_r:
                            done = True
                        elif event.key == pygame.K_UP:
                            self.speed = min(10, self.speed + 1)
                            print(f"Speed: {self.speed}x")
                        elif event.key == pygame.K_DOWN:
                            self.speed = max(1, self.speed - 1)
                            print(f"Speed: {self.speed}x")
                
                if self.step_mode and not self.step_requested and not self.paused:
                    self.render(state)
                    self.clock.tick(30)
                    continue
                
                if self.paused:
                    self.render(state)
                    self.clock.tick(30)
                    continue
                
                attacker_action = {}
                defender_action = {}
                action_info = {}
                
                if self.attacker:
                    try:
                        model_input = self.prepare_input(state)
                        with torch.no_grad():
                            attacker_strategy = self.attacker.get_strategy(model_input)
                            defender_strategy = self.defender.get_strategy(model_input)
                            
                            attacker_action = self.strategy_to_actions(
                                attacker_strategy, 
                                self.env.attacker_units, 
                                is_attacker=True
                            )
                            defender_action = self.strategy_to_actions(
                                defender_strategy, 
                                self.env.defender_units, 
                                is_attacker=False
                            )
                            
                            action_info = {
                                'attacker_lines': attacker_strategy.get('advance_lines'),
                                'defender_lines': defender_strategy.get('defense_lines')
                            }
                            
                    except Exception as e:
                        print(f"❌ Model error: {e}")
                        attacker_action = {}
                        defender_action = {}
                else:
                    for unit in self.env.attacker_units:
                        if unit.alive:
                            attacker_action[str(unit.id)] = {
                                'move': (unit.x + np.random.randint(-8, 8), 
                                        unit.y + np.random.randint(-8, 8))
                            }
                    for unit in self.env.defender_units:
                        if unit.alive:
                            defender_action[str(unit.id)] = {
                                'move': (unit.x + np.random.randint(-8, 8), 
                                        unit.y + np.random.randint(-8, 8))
                            }
                
                next_state, rewards, done, info = self.env.step(
                    attacker_action, defender_action
                )
                
                total_reward['attacker'] += rewards['attacker']
                total_reward['defender'] += rewards['defender']
                step_count += 1
                
                self.render(next_state, info, action_info)
                state = next_state
                
                if self.step_mode:
                    self.step_requested = False
                
                delay = int(300 / self.speed)
                pygame.time.wait(max(10, delay))
            
            self.episode_count += 1
            print(f"Episode {self.episode_count}: "
                  f"Attacker: {total_reward['attacker']:.0f}, "
                  f"Defender: {total_reward['defender']:.0f}, "
                  f"Steps: {step_count}")

def find_model():
    model_paths = [
        'models_saved/checkpoint_500/checkpoint_500.pt',
        'models_saved/final/checkpoint_56.pt',
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            return path
    
    pt_files = glob.glob('models_saved/**/*.pt', recursive=True)
    if pt_files:
        pt_files.sort(key=lambda f: os.path.getsize(f), reverse=True)
        return pt_files[0]
    
    return None

if __name__ == "__main__":
    model_path = find_model()
    
    if model_path:
        print(f"📦 Found model: {model_path}")
    else:
        print("⚠️ No model found")
    
    visualizer = StrategicVisualizer(model_path)
    visualizer.run()