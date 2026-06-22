# models/defender.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseStrategicModel
from typing import Dict, Tuple, Union
import numpy as np

class DefenderModel(BaseStrategicModel):
    """
    Защищающаяся модель - парирует атаки и строит оборону
    """
    def __init__(self, config):
        super().__init__(config, 'defender')
        
        self.defense_lines = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64)
        )
        
        self.threat_detector = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 16)
        )
        
        self.reserve_allocator = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 20)
        )
        
        self.counter_attack = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 8)
        )
        
        self.value_head = nn.Linear(256, 1)
        
    def forward(self, state: Union[Dict, torch.Tensor]):
        """Прямой проход"""
        features = super().forward(state)
        
        lines = self.defense_lines(features)
        lines = lines.view(-1, 32, 2)
        lines = torch.sigmoid(lines) * self.config.map_size
        
        threats = self.threat_detector(features)
        threats = threats.view(-1, 8, 2)
        threats = torch.sigmoid(threats) * self.config.map_size
        
        reserves = torch.softmax(self.reserve_allocator(features), dim=-1)
        counter = torch.softmax(self.counter_attack(features), dim=-1)
        value = self.value_head(features)
        
        return {
            'defense_lines': lines,
            'threats': threats,
            'reserves': reserves,
            'counter_attack': counter,
            'value': value,
            'features': features
        }
    
    def get_strategy(self, state: Union[Dict, torch.Tensor]) -> Dict:
        """Получение оборонительной стратегии"""
        self.eval()
        with torch.no_grad():
            output = self.forward(state)
            
            strategy = {
                'defense_lines': output['defense_lines'].cpu().numpy(),
                'threats': output['threats'].cpu().numpy(),
                'reserve_distribution': output['reserves'].cpu().numpy(),
                'counter_direction': torch.argmax(output['counter_attack']).item()
            }
            
            return strategy