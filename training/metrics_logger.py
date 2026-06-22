# training/metrics_logger.py
import json
from collections import deque
import numpy as np

class MetricsLogger:
    """Логгер для метрик обучения"""
    def __init__(self):
        self.episodes = []
        self.attacker_rewards = []
        self.defender_rewards = []
        self.attacker_wins = deque(maxlen=1000)
        self.defender_wins = deque(maxlen=1000)
        self.best_reward = -float('inf')
        
    def log_episode(self, results):
        """Логирование эпизода"""
        self.episodes.append(results['episode'])
        self.attacker_rewards.append(results['attacker_reward'])
        self.defender_rewards.append(results['defender_reward'])
        
        # Определение победителя
        if results['attacker_reward'] > results['defender_reward']:
            self.attacker_wins.append(1)
            self.defender_wins.append(0)
        else:
            self.attacker_wins.append(0)
            self.defender_wins.append(1)
        
        # Лучшая награда
        avg_reward = (results['attacker_reward'] + results['defender_reward']) / 2
        if avg_reward > self.best_reward:
            self.best_reward = avg_reward
    
    def get_win_rate(self, window=100):
        """Получение процента побед"""
        if len(self.attacker_wins) < window:
            window = len(self.attacker_wins)
        
        if window == 0:
            return {'attacker': 0, 'defender': 0}
        
        attacker_wins = list(self.attacker_wins)[-window:]
        defender_wins = list(self.defender_wins)[-window:]
        
        return {
            'attacker': sum(attacker_wins) / window * 100,
            'defender': sum(defender_wins) / window * 100
        }
    
    def get_best_reward(self):
        """Получение лучшей награды"""
        return self.best_reward
    
    def save(self, path):
        """Сохранение метрик"""
        data = {
            'episodes': list(self.episodes),
            'attacker_rewards': self.attacker_rewards,
            'defender_rewards': self.defender_rewards,
            'attacker_wins': list(self.attacker_wins),
            'defender_wins': list(self.defender_wins),
            'best_reward': self.best_reward
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)