# models/base_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union
import numpy as np

class BaseStrategicModel(nn.Module):
    """
    Базовая стратегическая модель
    """
    def __init__(self, config, model_type='attacker'):
        super().__init__()
        self.config = config
        self.model_type = model_type
        self.map_size = config.map_size
        self.n_units = config.max_units
        
        # Простой CNN для карты
        self.map_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Простой MLP для юнитов
        self.unit_encoder = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )
        
        # Объединяющий слой
        self.fusion = nn.Sequential(
            nn.Linear(256 + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
    def _prepare_input(self, state: Union[Dict, torch.Tensor]) -> Dict:
        """Подготовка входных данных - конвертация в тензоры"""
        if isinstance(state, torch.Tensor):
            return {'terrain': state}
        
        if not isinstance(state, dict):
            state = {}
        
        prepared = {}
        
        # Карта
        terrain = state.get('terrain')
        if terrain is not None:
            if isinstance(terrain, np.ndarray):
                if len(terrain.shape) == 2:
                    terrain = torch.FloatTensor(terrain).unsqueeze(0).unsqueeze(0)
                elif len(terrain.shape) == 3:
                    terrain = torch.FloatTensor(terrain).unsqueeze(0)
                else:
                    terrain = torch.zeros(1, 1, self.map_size, self.map_size)
            elif isinstance(terrain, torch.Tensor):
                if len(terrain.shape) == 2:
                    terrain = terrain.unsqueeze(0).unsqueeze(0)
                elif len(terrain.shape) == 3:
                    terrain = terrain.unsqueeze(0)
            else:
                terrain = torch.zeros(1, 1, self.map_size, self.map_size)
        else:
            terrain = torch.zeros(1, 1, self.map_size, self.map_size)
        
        # Убеждаемся что размерность правильная
        if terrain.shape[1] != 1:
            terrain = terrain.mean(dim=1, keepdim=True)
        
        prepared['terrain'] = terrain
        
        # Юниты
        units = state.get('units')
        if units is not None:
            if isinstance(units, np.ndarray):
                units = torch.FloatTensor(units)
            if len(units.shape) == 2:
                units = units.unsqueeze(0)
            if units.shape[2] != 10:
                if units.shape[2] < 10:
                    padding = torch.zeros(units.shape[0], units.shape[1], 10 - units.shape[2])
                    units = torch.cat([units, padding], dim=-1)
                else:
                    units = units[:, :, :10]
            prepared['units'] = units
        else:
            prepared['units'] = torch.zeros(1, self.n_units, 10)
        
        return prepared
    
    def forward(self, state: Union[Dict, torch.Tensor]):
        """
        Прямой проход
        """
        # Подготовка данных
        prepared = self._prepare_input(state)
        
        # Карта
        map_tensor = prepared['terrain']
        map_features = self.map_encoder(map_tensor)
        map_features = map_features.view(map_features.size(0), -1)
        
        # Юниты
        units_tensor = prepared['units']
        units_encoded = self.unit_encoder(units_tensor)
        units_pooled = units_encoded.mean(dim=1)
        
        # Объединение
        combined = torch.cat([map_features, units_pooled], dim=-1)
        features = self.fusion(combined)
        
        return features