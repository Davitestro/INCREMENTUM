# visualization/visualizer_laptop.py
import pygame
import torch
import numpy as np
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle

sys.path.append('..')
from config.base_config import BaseConfig
from environment.battle_env import BattleEnvironment
from models.attacker import AttackerModel
from models.defender import DefenderModel

class LaptopVisualizer:
    """
    Легкий визуализатор для ноутбука
    """
    def __init__(self, model_path=None):
        self.config = BaseConfig()
        self.config.map_size = 32  # Меньше для скорости
        
        # Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((1000, 800))
        pygame.display.set_caption("Strategic RL - Laptop Visualizer")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        
        # Среда
        self.env = BattleEnvironment(self.config)
        
        # Модели (если есть)
        self.attacker = None
        self.defender = None
        if model_path:
            self._load_models(model_path)
        
        # Режимы
        self.mode = 'auto'  # 'auto', 'manual', 'view'
        self.speed = 2
        self.paused = False
        self.show_info = True
        
        # Статистика
        self.reward_history = []
        self.episode_count = 0
        
    def _load_models(self, path):
        """Загрузка моделей"""
        try:
            checkpoint = torch.load(path, map_location='cpu')
            
            self.attacker = AttackerModel(self.config)
            self.attacker.load_state_dict(checkpoint['attacker'])
            self.attacker.eval()
            
            self.defender = DefenderModel(self.config)
            self.defender.load_state_dict(checkpoint['defender'])
            self.defender.eval()
            
            print(f"✅ Models loaded from {path}")
            return True
        except Exception as e:
            print(f"❌ Failed to load models: {e}")
            return False
    
    def run(self):
        """Главный цикл"""
        print("🎮 Strategic RL Visualizer")
        print("=" * 50)
        print("Controls:")
        print("  SPACE - Pause/Resume")
        print("  A - Attacker manual control")
        print("  D - Defender manual control")
        print("  V - View mode")
        print("  ↑↓ - Speed up/down")
        print("  R - Reset episode")
        print("  ESC - Quit")
        print("=" * 50)
        
        running = True
        
        while running:
            state = self.env.reset()
            done = False
            
            while not done and running:
                # Обработка событий
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_SPACE:
                            self.paused = not self.paused
                        elif event.key == pygame.K_r:
                            done = True
                        elif event.key == pygame.K_UP:
                            self.speed = min(10, self.speed + 1)
                        elif event.key == pygame.K_DOWN:
                            self.speed = max(1, self.speed - 1)
                        elif event.key == pygame.K_a:
                            self.mode = 'attacker_manual' if self.mode != 'attacker_manual' else 'auto'
                        elif event.key == pygame.K_d:
                            self.mode = 'defender_manual' if self.mode != 'defender_manual' else 'auto'
                        elif event.key == pygame.K_v:
                            self.mode = 'view' if self.mode != 'view' else 'auto'
                
                if self.paused:
                    self._render(state)
                    self.clock.tick(30)
                    continue
                
                # Получение стратегий
                if self.mode == 'auto':
                    attacker_strategy = self.attacker.get_strategy(state) if self.attacker else None
                    defender_strategy = self.defender.get_strategy(state) if self.defender else None
                elif self.mode == 'attacker_manual':
                    attacker_strategy = self._manual_strategy(state, 'attacker')
                    defender_strategy = self.defender.get_strategy(state) if self.defender else None
                elif self.mode == 'defender_manual':
                    attacker_strategy = self.attacker.get_strategy(state) if self.attacker else None
                    defender_strategy = self._manual_strategy(state, 'defender')
                else:  # view mode
                    attacker_strategy = {}
                    defender_strategy = {}
                
                # Шаг среды
                next_state, rewards, done, info = self.env.step(
                    attacker_strategy or {},
                    defender_strategy or {}
                )
                
                # Обновление статистики
                self.reward_history.append({
                    'attacker': rewards['attacker'],
                    'defender': rewards['defender']
                })
                
                # Рендер
                self._render(state, info, attacker_strategy, defender_strategy)
                
                state = next_state
                
                # Задержка для скорости
                pygame.time.wait(int(100 / self.speed))
            
            self.episode_count += 1
            print(f"Episode {self.episode_count} finished")
        
        pygame.quit()
        print("👋 Goodbye!")
    
    def _render(self, state, info=None, attacker_strategy=None, defender_strategy=None):
        """Рендер состояния"""
        self.screen.fill((30, 30, 30))
        
        map_size = self.config.map_size
        cell_size = 600 // map_size
        
        # Рисуем карту
        terrain = state.get('terrain', np.zeros((map_size, map_size, 3)))
        
        for y in range(map_size):
            for x in range(map_size):
                rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
                
                # Цвет по высоте
                height = terrain[y, x, 0] if len(terrain.shape) > 2 else terrain[y, x]
                color = (50 + int(height * 100), 50 + int(height * 80), 50 + int(height * 60))
                pygame.draw.rect(self.screen, color, rect)
                
                # Сетка
                pygame.draw.rect(self.screen, (60, 60, 70), rect, 1)
        
        # Рисуем юниты
        for unit in self.env.attacker_units:
            if unit.hp > 0:
                self._draw_unit(unit, (0, 200, 100), cell_size)
        
        for unit in self.env.defender_units:
            if unit.hp > 0:
                self._draw_unit(unit, (200, 50, 50), cell_size)
        
        # Рисуем стратегические линии
        if attacker_strategy and 'advance_lines' in attacker_strategy:
            self._draw_strategy_lines(attacker_strategy['advance_lines'], (0, 255, 0), cell_size)
        
        if defender_strategy and 'defense_lines' in defender_strategy:
            self._draw_strategy_lines(defender_strategy['defense_lines'], (255, 0, 0), cell_size)
        
        # Информация справа
        if self.show_info:
            self._render_info_panel(info)
        
        pygame.display.flip()
    
    def _draw_unit(self, unit, color, cell_size):
        """Рисование юнита"""
        x = int(unit.x * cell_size + cell_size // 2)
        y = int(unit.y * cell_size + cell_size // 2)
        radius = cell_size // 3
        
        pygame.draw.circle(self.screen, color, (x, y), radius)
        pygame.draw.circle(self.screen, (255, 255, 255), (x, y), radius, 1)
        
        # HP
        hp_text = self.font.render(f"{int(unit.hp)}", True, (255, 255, 255))
        self.screen.blit(hp_text, (x - 10, y - 10))
    
    def _draw_strategy_lines(self, lines, color, cell_size):
        """Рисование стратегических линий"""
        if len(lines.shape) == 3:
            lines = lines[0]  # Берем первую линию
        
        points = []
        for point in lines:
            x = int(point[0] * cell_size)
            y = int(point[1] * cell_size)
            points.append((x, y))
        
        if len(points) > 1:
            pygame.draw.lines(self.screen, color, False, points, 3)
            
            # Стрелки
            for i in range(0, len(points) - 5, 5):
                p1 = points[i]
                p2 = points[i + 5]
                self._draw_arrow(p1, p2, color)
    
    def _draw_arrow(self, start, end, color):
        """Рисование стрелки"""
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = np.sqrt(dx**2 + dy**2)
        
        if length > 20:
            angle = np.arctan2(dy, dx)
            arrow_len = 10
            
            p1 = (end[0] - arrow_len * np.cos(angle - 0.4),
                  end[1] - arrow_len * np.sin(angle - 0.4))
            p2 = (end[0] - arrow_len * np.cos(angle + 0.4),
                  end[1] - arrow_len * np.sin(angle + 0.4))
            
            pygame.draw.line(self.screen, color, end, p1, 2)
            pygame.draw.line(self.screen, color, end, p2, 2)
    
    def _render_info_panel(self, info):
        """Рендер панели информации"""
        x = 620
        y = 10
        line_height = 25
        
        info_data = [
            f"Episode: {self.episode_count}",
            f"Mode: {self.mode}",
            f"Speed: {self.speed}x",
            f"Paused: {self.paused}",
            "",
            f"Attacker units: {sum(1 for u in self.env.attacker_units if u.hp > 0)}",
            f"Defender units: {sum(1 for u in self.env.defender_units if u.hp > 0)}",
            "",
            f"Attacker reward: {self.reward_history[-1]['attacker'] if self.reward_history else 0:.1f}",
            f"Defender reward: {self.reward_history[-1]['defender'] if self.reward_history else 0:.1f}"
        ]
        
        for line in info_data:
            text = self.font.render(line, True, (255, 255, 255))
            self.screen.blit(text, (x, y))
            y += line_height
    
    def _manual_strategy(self, state, team):
        """Ручное управление"""
        # TODO: Реализовать ручное управление через мышь
        return {}

if __name__ == "__main__":
    # Поиск модели
    model_path = None
    models_dir = Path("models_saved")
    if models_dir.exists():
        checkpoints = list(models_dir.glob("*.pt"))
        if checkpoints:
            latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
            model_path = str(latest)
            print(f"📦 Found model: {model_path}")
    
    visualizer = LaptopVisualizer(model_path)
    visualizer.run()