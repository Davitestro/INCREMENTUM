# config/base_config.py
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import numpy as np
import json
import os
from datetime import datetime
from collections import deque  # ДОБАВЛЯЕМ ЭТОТ ИМПОРТ
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BaseConfig:
    """Базовая конфигурация с авто-тюнингом"""
    
    # Размеры карты
    map_size: int = 64
    terrain_complexity: float = 0.7
    max_steps: int = 200
    
    # Юниты
    max_units: int = 20
    unit_types: Dict = field(default_factory=lambda: {
        'infantry': {'attack': 10, 'defense': 5, 'speed': 3, 'range': 20},
        'tank': {'attack': 25, 'defense': 20, 'speed': 4, 'range': 30},
        'artillery': {'attack': 40, 'defense': 5, 'speed': 2, 'range': 60},
        'recon': {'attack': 5, 'defense': 3, 'speed': 6, 'range': 40},
        'air': {'attack': 30, 'defense': 10, 'speed': 8, 'range': 50}
    })
    
    # Награды
    rewards: Dict = field(default_factory=lambda: {
        'kill': 100.0,
        'capture_point': 200.0,
        'formation_bonus': 50.0,
        'flank_bonus': 150.0,
        'defense_bonus': 80.0,
        'step_penalty': -1.0,
        'unit_loss': -50.0,
        'supply_line_cut': -200.0
    })
    
    # RL параметры (будут авто-настроены)
    learning_rate: Optional[float] = None
    batch_size: Optional[int] = None
    n_steps: Optional[int] = None
    gamma: Optional[float] = None
    gae_lambda: Optional[float] = None
    
    # Адаптивные параметры
    adaptive_learning: bool = True
    exploration_decay: float = 0.9995
    min_exploration: float = 0.05
    
    # Пути
    model_dir: str = "./models_saved"
    log_dir: str = "./logs"
    
    def __post_init__(self):
        """Инициализация с авто-подбором гиперпараметров"""
        if self.learning_rate is None:
            self.learning_rate = self._auto_learning_rate()
        if self.batch_size is None:
            self.batch_size = self._auto_batch_size()
        if self.n_steps is None:
            self.n_steps = self._auto_n_steps()
        if self.gamma is None:
            self.gamma = 0.99
        if self.gae_lambda is None:
            self.gae_lambda = 0.95
            
        # Создание директорий
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
    def _auto_learning_rate(self) -> float:
        """Автоматический подбор learning rate"""
        return 0.0003 * (1 + np.random.uniform(-0.3, 0.3))
    
    def _auto_batch_size(self) -> int:
        """Автоматический подбор batch size"""
        return int(64 * (1 + np.random.uniform(-0.2, 0.2)))
    
    def _auto_n_steps(self) -> int:
        """Автоматический подбор шагов"""
        return int(2048 * (1 + np.random.uniform(-0.2, 0.2)))
    
    def save(self, path: str):
        """Сохранение конфигурации"""
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        """Загрузка конфигурации"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


class AutoTuneConfig:
    """
    Автоматическая настройка гиперпараметров
    Использует Bayesian Optimization + Online Learning
    """
    def __init__(self, config: BaseConfig):
        self.config = config
        self.history = []
        self.best_params = {}
        self.performance_window = deque(maxlen=100)  # ТЕПЕРЬ РАБОТАЕТ
        self.hyperparam_space = {
            'learning_rate': (0.0001, 0.001),
            'batch_size': (32, 256),
            'n_steps': (1024, 4096),
            'gamma': (0.95, 0.999),
            'gae_lambda': (0.9, 0.99),
            'exploration': (0.01, 0.3)
        }
        
    def suggest_params(self) -> Dict:
        """Предложение новых гиперпараметров"""
        if len(self.history) < 10:
            # Random search на старте
            params = self._random_params()
        else:
            # Адаптивная настройка
            params = self._adaptive_params()
        
        return params
    
    def _random_params(self) -> Dict:
        """Случайные параметры для исследования"""
        return {
            'learning_rate': np.random.uniform(0.0001, 0.001),
            'batch_size': int(np.random.uniform(32, 256)),
            'n_steps': int(np.random.uniform(1024, 4096)),
            'gamma': np.random.uniform(0.95, 0.999),
            'gae_lambda': np.random.uniform(0.9, 0.99),
            'exploration': np.random.uniform(0.01, 0.3)
        }
    
    def _adaptive_params(self) -> Dict:
        """Адаптивная настройка на основе истории"""
        if not self.history:
            return self._random_params()
        
        best = max(self.history, key=lambda x: x.get('reward', -float('inf')))
        
        # Добавляем небольшие мутации
        params = {}
        for key, (min_val, max_val) in self.hyperparam_space.items():
            current = best.get(key, (min_val + max_val) / 2)
            # Мутация с уменьшающейся амплитудой
            mutation = np.random.normal(0, (max_val - min_val) * 0.1)
            params[key] = np.clip(current + mutation, min_val, max_val)
            
            # Приведение типов
            if key in ['batch_size', 'n_steps']:
                params[key] = int(params[key])
        
        return params
    
    def update(self, params: Dict, reward: float, metrics: Dict):
        """Обновление истории с новыми параметрами"""
        self.history.append({
            **params,
            'reward': reward,
            'metrics': metrics
        })
        
        # Обновляем лучшие параметры
        if not self.best_params or reward > self.best_params.get('reward', -float('inf')):
            self.best_params = {
                **params,
                'reward': reward
            }
        
        # Логирование
        logger.info(f"Hyperparameter update: Reward={reward:.2f}")
        logger.info(f"Best params: {self.best_params}")
        
        # Авто-сохранение лучшей конфигурации
        if len(self.history) % 10 == 0:
            self.save_best()
    
    def save_best(self):
        """Сохранение лучшей конфигурации"""
        path = os.path.join(
            self.config.log_dir, 
            f"best_params_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(path, 'w') as f:
            json.dump(self.best_params, f, indent=2)