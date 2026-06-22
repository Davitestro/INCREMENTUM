# environment/unit_controller.py
import numpy as np
from typing import List, Dict, Tuple, Optional
import random
from dataclasses import dataclass

@dataclass
class MilitaryUnit:
    """Военный юнит"""
    id: int
    unit_type: str
    team: str  # 'attacker' or 'defender'
    x: float
    y: float
    hp: float = 100.0
    max_hp: float = 100.0
    attack: float = 10.0
    defense: float = 5.0
    speed: float = 5.0
    range: float = 50.0
    morale: float = 1.0
    fatigue: float = 0.0
    ammo: float = 100.0
    alive: bool = True
    formation: str = 'line'
    orders: List[str] = None
    
    def __post_init__(self):
        if self.orders is None:
            self.orders = []

class UnitController:
    """
    Контроллер для управления юнитами
    """
    def __init__(self, config):
        self.config = config
        self.unit_id_counter = 0
        
    def initialize_units(self, team: str, n_units: int, terrain: np.ndarray) -> List[MilitaryUnit]:
        """
        Инициализация юнитов на карте
        """
        units = []
        map_size = self.config.map_size
        
        # Определяем зону размещения
        if team == 'attacker':
            x_range = (0, map_size // 3)
        else:
            x_range = (2 * map_size // 3, map_size)
        
        # Размещаем юниты
        for i in range(n_units):
            # Позиция с учетом рельефа
            x = random.randint(x_range[0], x_range[1])
            y = random.randint(0, map_size - 1)
            
            # Корректируем позицию на карте высот
            if len(terrain.shape) == 2:
                # Ищем ближайшую подходящую позицию
                for dy in range(-3, 4):
                    for dx in range(-3, 4):
                        nx = max(0, min(map_size - 1, x + dx))
                        ny = max(0, min(map_size - 1, y + dy))
                        if terrain[ny, nx] < 0.8:  # Не на горе
                            x, y = nx, ny
                            break
                    if terrain[y, x] < 0.8:
                        break
            
            # Создаем юнит
            unit_type = self._choose_unit_type(team)
            unit = MilitaryUnit(
                id=self.unit_id_counter,
                unit_type=unit_type,
                team=team,
                x=float(x),
                y=float(y),
                **self._get_unit_stats(unit_type)
            )
            self.unit_id_counter += 1
            units.append(unit)
        
        return units
    
    def _choose_unit_type(self, team: str) -> str:
        """Выбор типа юнита"""
        types = ['infantry', 'tank', 'artillery', 'recon']
        
        if team == 'attacker':
            weights = [0.3, 0.3, 0.2, 0.2]  # Больше танков
        else:
            weights = [0.4, 0.2, 0.2, 0.2]  # Больше пехоты
        
        return random.choices(types, weights=weights)[0]
    
    def _get_unit_stats(self, unit_type: str) -> Dict:
        """Получение характеристик юнита"""
        stats = {
            'infantry': {'attack': 10, 'defense': 8, 'speed': 3, 'range': 20, 'max_hp': 100},
            'tank': {'attack': 25, 'defense': 20, 'speed': 4, 'range': 30, 'max_hp': 150},
            'artillery': {'attack': 40, 'defense': 5, 'speed': 2, 'range': 60, 'max_hp': 80},
            'recon': {'attack': 5, 'defense': 3, 'speed': 6, 'range': 40, 'max_hp': 60}
        }
        return stats.get(unit_type, stats['infantry'])
    
    def move_unit(self, unit: MilitaryUnit, target_x: float, target_y: float):
        """Перемещение юнита"""
        dx = target_x - unit.x
        dy = target_y - unit.y
        dist = np.sqrt(dx**2 + dy**2)
        
        if dist > 0:
            # Ограничение скорости
            move_dist = min(unit.speed, dist)
            unit.x += (dx / dist) * move_dist
            unit.y += (dy / dist) * move_dist
            
            # Увеличиваем усталость
            unit.fatigue += 0.01 * (move_dist / unit.speed)
            unit.fatigue = min(1.0, unit.fatigue)
    
    def attack(self, unit: MilitaryUnit, target: MilitaryUnit) -> float:
        """Атака юнита"""
        if unit.ammo <= 0:
            return 0.0
        
        # Проверка расстояния
        dist = np.sqrt((unit.x - target.x)**2 + (unit.y - target.y)**2)
        if dist > unit.range:
            return 0.0
        
        # Расчет урона
        damage = unit.attack * (1 - dist / unit.range) * unit.morale * (1 - unit.fatigue * 0.3)
        damage *= (1 - target.defense / 100)
        damage = max(0, damage + np.random.normal(0, 2))
        
        # Тратим боеприпасы
        unit.ammo -= 5
        unit.ammo = max(0, unit.ammo)
        
        return damage
    
    def update_unit_state(self, unit: MilitaryUnit):
        """Обновление состояния юнита"""
        # Восстановление морали
        if unit.morale < 1.0:
            unit.morale += 0.01
        
        # Восстановление от усталости
        if unit.fatigue > 0:
            unit.fatigue -= 0.005
        
        # Проверка на выживание
        if unit.hp <= 0:
            unit.alive = False
            unit.hp = 0