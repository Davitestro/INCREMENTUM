# models/attacker.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseStrategicModel
from typing import Dict, Tuple, Union
import numpy as np

class AttackerModel(BaseStrategicModel):
    """
    Атакующая модель - генерирует стратегию наступления
    """
    def __init__(self, config):
        super().__init__(config, 'attacker')
        
        # Стратегические головы
        self.line_generator = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64)
        )
        
        self.direction_predictor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 8)
        )
        
        self.formation_selector = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
        
        self.force_distributor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 20)
        )
        
        self.value_head = nn.Linear(256, 1)
        
    def forward(self, state: Union[Dict, torch.Tensor]):
        """Прямой проход"""
        features = super().forward(state)
        
        lines = self.line_generator(features)
        lines = lines.view(-1, 32, 2)
        lines = torch.sigmoid(lines) * self.config.map_size
        
        directions = torch.softmax(self.direction_predictor(features), dim=-1)
        formation = torch.softmax(self.formation_selector(features), dim=-1)
        force_dist = torch.softmax(self.force_distributor(features), dim=-1)
        value = self.value_head(features)
        
        return {
            'lines': lines,
            'directions': directions,
            'formation': formation,
            'force_distribution': force_dist,
            'value': value,
            'features': features
        }
    
    def get_strategy(self, state: Union[Dict, torch.Tensor]) -> Dict:
        """Получение стратегии для применения"""
        self.eval()
        with torch.no_grad():
            output = self.forward(state)
            
            strategy = {
                'advance_lines': output['lines'].cpu().numpy(),
                'main_direction': torch.argmax(output['directions']).item(),
                'formation': torch.argmax(output['formation']).item(),
                'force_distribution': output['force_distribution'].cpu().numpy()
            }
            
            return strategy