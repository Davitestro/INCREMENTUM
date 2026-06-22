# environment/terrain_generator.py
import numpy as np
from typing import List, Tuple, Optional
from shapely.geometry import Polygon, Point
import random

class TerrainGenerator:
    """
    Генератор ландшафта для стратегической карты
    """
    def __init__(self, config):
        self.config = config
        self.map_size = config.map_size
        self.terrain_map = None
        self.obstacles = []
        self.defense_bonuses = {}
        
    def generate(self) -> np.ndarray:
        """
        Генерация карты высот с препятствиями
        Returns: [map_size, map_size] массив высот 0-1
        """
        # Базовый рельеф (шум Перлина упрощённо)
        terrain = np.zeros((self.map_size, self.map_size))
        
        # Создаем несколько холмов и впадин
        for _ in range(10):
            cx = random.randint(0, self.map_size - 1)
            cy = random.randint(0, self.map_size - 1)
            radius = random.randint(5, 15)
            height = random.uniform(0.3, 0.8)
            
            for y in range(max(0, cy - radius), min(self.map_size, cy + radius)):
                for x in range(max(0, cx - radius), min(self.map_size, cx + radius)):
                    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                    if dist < radius:
                        terrain[y, x] += height * (1 - dist / radius)
        
        # Нормализация
        terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min() + 0.001)
        terrain = np.clip(terrain, 0, 1)
        
        # Добавляем препятствия (водные преграды, горы)
        self.obstacles = self._generate_obstacles()
        
        self.terrain_map = terrain
        self._calculate_defense_bonuses()
        
        return terrain
    
    def _generate_obstacles(self) -> List:
        """Генерация препятствий"""
        obstacles = []
        
        # Реки
        for _ in range(2):
            start_x = random.randint(0, self.map_size // 4)
            start_y = random.randint(0, self.map_size)
            end_x = random.randint(3 * self.map_size // 4, self.map_size)
            end_y = random.randint(0, self.map_size)
            
            # Создаем полигон реки (кривая линия с шириной)
            points = []
            for t in np.linspace(0, 1, 10):
                x = start_x + (end_x - start_x) * t + np.sin(t * 5) * 10
                y = start_y + (end_y - start_y) * t + np.cos(t * 3) * 10
                points.append((x, y))
            
            # Добавляем ширину
            river_points = []
            for i, (x, y) in enumerate(points):
                if i < len(points) - 1:
                    nx = points[i+1][1] - y
                    ny = x - points[i+1][0]
                    norm = np.sqrt(nx**2 + ny**2)
                    if norm > 0:
                        nx /= norm
                        ny /= norm
                        river_points.append((x + nx * 3, y + ny * 3))
                        river_points.append((x - nx * 3, y - ny * 3))
            
            if len(river_points) > 2:
                obstacles.append(Polygon(river_points))
        
        # Леса
        for _ in range(3):
            cx = random.randint(0, self.map_size)
            cy = random.randint(0, self.map_size)
            radius = random.randint(8, 20)
            
            points = []
            for i in range(12):
                angle = i * 2 * np.pi / 12
                r = radius * (0.8 + random.random() * 0.4)
                points.append((cx + r * np.cos(angle), cy + r * np.sin(angle)))
            
            obstacles.append(Polygon(points))
        
        # Горы
        for _ in range(2):
            cx = random.randint(0, self.map_size)
            cy = random.randint(0, self.map_size)
            radius = random.randint(5, 10)
            
            points = []
            for i in range(8):
                angle = i * 2 * np.pi / 8
                r = radius * (0.7 + random.random() * 0.3)
                points.append((cx + r * np.cos(angle), cy + r * np.sin(angle)))
            
            obstacles.append(Polygon(points))
        
        return obstacles
    
    def _calculate_defense_bonuses(self):
        """Расчет бонусов обороны для каждой клетки"""
        self.defense_bonuses = {}
        for y in range(self.map_size):
            for x in range(self.map_size):
                bonus = 1.0
                
                # Высота дает преимущество
                bonus += self.terrain_map[y, x] * 0.3
                
                # Проверка на препятствия
                point = Point(x, y)
                for obs in self.obstacles:
                    if obs.contains(point):
                        bonus += 0.2
                        break
                
                self.defense_bonuses[(x, y)] = bonus
    
    def get_defense_bonus(self, x: float, y: float) -> float:
        """Получение бонуса обороны для позиции"""
        ix, iy = int(x), int(y)
        if 0 <= ix < self.map_size and 0 <= iy < self.map_size:
            return self.defense_bonuses.get((ix, iy), 1.0)
        return 1.0
    
    def get_obstacles(self) -> List:
        """Получение списка препятствий"""
        return self.obstacles
    
    def get_height(self, x: float, y: float) -> float:
        """Получение высоты в точке"""
        ix, iy = int(x), int(y)
        if 0 <= ix < self.map_size and 0 <= iy < self.map_size:
            return self.terrain_map[iy, ix]
        return 0.0