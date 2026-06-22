# environment/battle_env.py
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
import random
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union

from environment.terrain_generator import TerrainGenerator
from environment.unit_controller import UnitController, MilitaryUnit
from config.base_config import BaseConfig

class BattleEnvironment:
    """
    Среда для стратегической битвы
    """
    def __init__(self, config: BaseConfig, mode: str = 'train'):
        self.config = config
        self.mode = mode
        self.map_size = config.map_size
        
        # Генераторы
        self.terrain_gen = TerrainGenerator(config)
        self.unit_ctrl = UnitController(config)
        
        # Состояние
        self.attacker_units: List[MilitaryUnit] = []
        self.defender_units: List[MilitaryUnit] = []
        self.terrain_map = None
        self.obstacles = []
        self.front_line = []
        self.controlled_areas = {}
        self.objectives = []
        
        # Статистика
        self.step_count = 0
        self.casualties = {'attacker': 0, 'defender': 0}
        self.total_reward = {'attacker': 0, 'defender': 0}
        
        self.reset()
    
    def reset(self) -> Dict:
        """Сброс среды"""
        # Генерация карты
        self.terrain_map = self.terrain_gen.generate()
        self.obstacles = self.terrain_gen.get_obstacles()
        
        # Генерация юнитов
        self.attacker_units = self.unit_ctrl.initialize_units(
            team='attacker',
            n_units=self.config.max_units,
            terrain=self.terrain_map
        )
        
        self.defender_units = self.unit_ctrl.initialize_units(
            team='defender',
            n_units=self.config.max_units,
            terrain=self.terrain_map
        )
        
        # Начальная линия фронта
        self.front_line = self._calculate_initial_front_line()
        
        # Цели
        self.objectives = self._generate_objectives()
        
        # Зоны контроля
        self.controlled_areas = {
            'attacker': self._calculate_control_zone('attacker'),
            'defender': self._calculate_control_zone('defender')
        }
        
        self.step_count = 0
        self.total_reward = {'attacker': 0, 'defender': 0}
        self.casualties = {'attacker': 0, 'defender': 0}
        
        return self.get_state()
    
    def step(self, attacker_actions: Dict, defender_actions: Dict) -> Tuple[Dict, Dict, bool, Dict]:
        """
        Шаг симуляции
        """
        self.step_count += 1
        
        # Применяем действия
        attacker_reward = self._apply_actions(
            self.attacker_units, 
            attacker_actions, 
            self.defender_units
        )
        
        defender_reward = self._apply_actions(
            self.defender_units, 
            defender_actions, 
            self.attacker_units
        )
        
        # Симуляция боя
        battle_results = self._simulate_battle()
        
        # Обновление наград
        self.total_reward['attacker'] += attacker_reward + battle_results['attacker_reward']
        self.total_reward['defender'] += defender_reward + battle_results['defender_reward']
        
        # Обновление линии фронта
        self.front_line = self._update_front_line()
        
        # Обновление зон контроля
        self.controlled_areas = {
            'attacker': self._calculate_control_zone('attacker'),
            'defender': self._calculate_control_zone('defender')
        }
        
        # Проверка завершения эпизода
        done = self._check_done()
        
        # Состояние для следующего шага
        next_state = self.get_state()
        
        info = {
            'attacker_reward': attacker_reward,
            'defender_reward': defender_reward,
            'battle_results': battle_results,
            'front_line': self.front_line,
            'controlled_areas': self.controlled_areas,
            'casualties': self.casualties,
            'step_count': self.step_count
        }
        
        return next_state, self.total_reward, done, info
    
    def get_state(self) -> Dict:
        """
        Возвращает полное состояние для моделей
        """
        # Создаем карту состояний
        state = {
            'terrain': self.terrain_map.copy() if self.terrain_map is not None else np.zeros((self.map_size, self.map_size)),
            'step': self.step_count,
            'attacker_count': sum(1 for u in self.attacker_units if u.alive),
            'defender_count': sum(1 for u in self.defender_units if u.alive)
        }
        
        # Добавляем информацию о юнитах в виде тензора
        units_data = self._get_units_tensor()
        state['units'] = units_data
        
        # Линия фронта
        state['front_line'] = self.front_line
        
        # Объективы
        state['objectives'] = self.objectives
        
        return state
    
    def _get_units_tensor(self) -> np.ndarray:
        """
        Создает тензор с информацией о юнитах
        Формат: [n_units, features]
        features: x, y, hp, team, type
        """
        all_units = self.attacker_units + self.defender_units
        n_units = len(all_units)
        
        if n_units == 0:
            return np.zeros((self.config.max_units, 10))
        
        features = []
        for unit in all_units:
            # Кодируем тип юнита one-hot
            type_onehot = [0, 0, 0, 0, 0]  # infantry, tank, artillery, recon, air
            type_map = {'infantry': 0, 'tank': 1, 'artillery': 2, 'recon': 3, 'air': 4}
            if unit.unit_type in type_map:
                type_onehot[type_map[unit.unit_type]] = 1
            
            unit_features = [
                unit.x / self.map_size,  # нормализованная x
                unit.y / self.map_size,  # нормализованная y
                unit.hp / unit.max_hp,   # нормализованный HP
                1.0 if unit.team == 'attacker' else 0.0,  # team
                *type_onehot,  # тип юнита one-hot
                unit.morale,
                unit.fatigue,
                unit.ammo / 100.0
            ]
            features.append(unit_features)
        
        # Дополняем до max_units
        while len(features) < self.config.max_units:
            features.append([0.0] * 10)
        
        return np.array(features[:self.config.max_units])
    
    def _apply_actions(self, units: List[MilitaryUnit], actions: Dict, enemy_units: List[MilitaryUnit]) -> float:
        """Применение действий к юнитам"""
        reward = 0.0
        
        for unit in units:
            if not unit.alive:
                continue
            
            action = actions.get(str(unit.id), {})
            
            # Перемещение
            if 'move' in action:
                target = action['move']
                if isinstance(target, (tuple, list)) and len(target) == 2:
                    self.unit_ctrl.move_unit(unit, target[0], target[1])
                    reward += 0.1
            
            # Атака
            if 'attack' in action:
                target_id = action['attack']
                for enemy in enemy_units:
                    if enemy.id == target_id and enemy.alive:
                        damage = self.unit_ctrl.attack(unit, enemy)
                        enemy.hp -= damage
                        reward += damage / 10
                        
                        if enemy.hp <= 0:
                            enemy.alive = False
                            reward += 10.0
                            self.casualties[enemy.team] += 1
                        break
        
        return reward
    
    def _simulate_battle(self) -> Dict:
        """Симуляция боя между юнитами"""
        rewards = {'attacker_reward': 0.0, 'defender_reward': 0.0}
        
        # Находим юниты в зоне боя
        for attacker in self.attacker_units:
            if not attacker.alive:
                continue
                
            for defender in self.defender_units:
                if not defender.alive:
                    continue
                
                # Проверка расстояния
                dist = np.sqrt((attacker.x - defender.x)**2 + (attacker.y - defender.y)**2)
                
                if dist < attacker.range + defender.range:
                    # Бой
                    damage_attacker = self.unit_ctrl.attack(attacker, defender)
                    damage_defender = self.unit_ctrl.attack(defender, attacker)
                    
                    defender.hp -= damage_attacker
                    attacker.hp -= damage_defender
                    
                    if defender.hp <= 0:
                        defender.alive = False
                        rewards['attacker_reward'] += self.config.rewards.get('kill', 100.0)
                        self.casualties['defender'] += 1
                    
                    if attacker.hp <= 0:
                        attacker.alive = False
                        rewards['defender_reward'] += self.config.rewards.get('kill', 100.0)
                        self.casualties['attacker'] += 1
        
        return rewards
    
    def _check_done(self) -> bool:
        """Проверка окончания эпизода"""
        attacker_alive = sum(1 for u in self.attacker_units if u.alive)
        defender_alive = sum(1 for u in self.defender_units if u.alive)
        
        if attacker_alive == 0 or defender_alive == 0:
            return True
        
        if self.step_count >= self.config.max_steps:
            return True
        
        return False
    
    def _calculate_initial_front_line(self) -> List[Tuple[float, float]]:
        """Расчет начальной линии фронта"""
        line = []
        for y in range(0, self.map_size, 5):
            x = self.map_size // 2 + np.random.normal(0, 5)
            line.append((x, float(y)))
        return line
    
    def _generate_objectives(self) -> List[Dict]:
        """Генерация стратегических целей"""
        objectives = []
        
        # Центральная точка
        cx, cy = self.map_size // 2, self.map_size // 2
        objectives.append({
            'type': 'capture',
            'x': cx,
            'y': cy,
            'radius': 10,
            'value': 200
        })
        
        # Фланговые точки
        for dx, dy in [(-20, 0), (20, 0), (0, -20), (0, 20)]:
            objectives.append({
                'type': 'strategic',
                'x': cx + dx,
                'y': cy + dy,
                'radius': 8,
                'value': 100
            })
        
        return objectives
    
    def _calculate_control_zone(self, team: str) -> Any:
        """Расчет зоны контроля"""
        units = self.attacker_units if team == 'attacker' else self.defender_units
        
        if not units:
            return Polygon()
        
        points = []
        for unit in units:
            if unit.alive:
                points.append((unit.x, unit.y))
        
        if len(points) < 3:
            return Polygon()
        
        # Создаем выпуклую оболочку
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(points)
            hull_points = [points[i] for i in hull.vertices]
            return Polygon(hull_points)
        except:
            return Polygon()
    
    def _update_front_line(self) -> List[Tuple[float, float]]:
        """Обновление линии фронта"""
        line = []
        
        attacker_x = np.mean([u.x for u in self.attacker_units if u.alive]) if self.attacker_units else 0
        defender_x = np.mean([u.x for u in self.defender_units if u.alive]) if self.defender_units else self.map_size
        
        if attacker_x and defender_x:
            mid_x = (attacker_x + defender_x) / 2
            for y in range(0, self.map_size, 5):
                line.append((mid_x + np.random.normal(0, 3), float(y)))
        
        return line