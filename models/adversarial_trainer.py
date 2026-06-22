# models/adversarial_trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import time
import random
from pathlib import Path
import logging

from config.base_config import BaseConfig, AutoTuneConfig
from environment.battle_env import BattleEnvironment
from models.attacker import AttackerModel
from models.defender import DefenderModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdversarialTrainer:
    """
    Противоборствующий тренер для двух моделей
    """
    def __init__(self, config: BaseConfig):
        self.config = config
        self.auto_tune = AutoTuneConfig(config)
        
        # Модели
        self.attacker = AttackerModel(config)
        self.defender = DefenderModel(config)
        
        # Оптимизаторы
        self.attacker_optimizer = optim.Adam(
            self.attacker.parameters(),
            lr=config.learning_rate
        )
        self.defender_optimizer = optim.Adam(
            self.defender.parameters(),
            lr=config.learning_rate
        )
        
        # Среда
        self.env = BattleEnvironment(config)
        
        # Память опыта
        self.attacker_memory = deque(maxlen=100000)
        self.defender_memory = deque(maxlen=100000)
        
        # Статистика
        self.metrics = {
            'attacker_rewards': [],
            'defender_rewards': [],
            'attacker_loss': [],
            'defender_loss': [],
            'attacker_win_rate': [],
            'defender_win_rate': [],
            'episode_length': []
        }
        
        # Устройство
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            logger.info("Using CPU")
        
        self.attacker.to(self.device)
        self.defender.to(self.device)
        
        self.episode_count = 0
        
        logger.info(f"AdversarialTrainer initialized on {self.device}")
        logger.info(f"Attacker params: {sum(p.numel() for p in self.attacker.parameters())}")
        logger.info(f"Defender params: {sum(p.numel() for p in self.defender.parameters())}")
    
    def _prepare_model_input(self, state: Dict) -> Dict:
        """Подготовка входа для модели"""
        try:
            if state is None:
                state = {}
            
            # Подготовка карты
            terrain = state.get('terrain')
            if terrain is not None:
                if isinstance(terrain, np.ndarray):
                    if len(terrain.shape) == 2:
                        terrain = torch.FloatTensor(terrain).unsqueeze(0).unsqueeze(0)
                    elif len(terrain.shape) == 3:
                        terrain = torch.FloatTensor(terrain).unsqueeze(0)
                    else:
                        terrain = torch.zeros(1, 1, self.config.map_size, self.config.map_size)
                elif isinstance(terrain, torch.Tensor):
                    if len(terrain.shape) == 2:
                        terrain = terrain.unsqueeze(0).unsqueeze(0)
                    elif len(terrain.shape) == 3:
                        terrain = terrain.unsqueeze(0)
                else:
                    terrain = torch.zeros(1, 1, self.config.map_size, self.config.map_size)
            else:
                terrain = torch.zeros(1, 1, self.config.map_size, self.config.map_size)
            
            # Проверка размерности
            if terrain.dim() == 2:
                terrain = terrain.unsqueeze(0).unsqueeze(0)
            elif terrain.dim() == 3:
                terrain = terrain.unsqueeze(0)
            
            if terrain.shape[1] != 1:
                terrain = terrain.mean(dim=1, keepdim=True)
            
            terrain = terrain.to(self.device)
            
            # Подготовка юнитов
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
            else:
                units = torch.zeros(1, self.config.max_units, 10)
            
            units = units.to(self.device)
            
            return {'terrain': terrain, 'units': units}
            
        except Exception as e:
            logger.warning(f"Error preparing input: {e}")
            return {
                'terrain': torch.zeros(1, 1, self.config.map_size, self.config.map_size).to(self.device),
                'units': torch.zeros(1, self.config.max_units, 10).to(self.device)
            }
    
    def train_episode(self) -> Dict:
        """Обучение одного эпизода"""
        state = self.env.reset()
        done = False
        episode_rewards = {'attacker': 0, 'defender': 0}
        episode_steps = 0
        
        while not done and episode_steps < self.config.max_steps:
            try:
                # Получение действий от моделей
                model_input = self._prepare_model_input(state)
                
                if self.attacker:
                    attacker_action = self.attacker.get_strategy(model_input)
                else:
                    attacker_action = {}
                    
                if self.defender:
                    defender_action = self.defender.get_strategy(model_input)
                else:
                    defender_action = {}
                    
            except Exception as e:
                logger.warning(f"Model error: {e}")
                attacker_action = {}
                defender_action = {}
            
            try:
                # Шаг среды
                next_state, rewards, done, info = self.env.step(
                    attacker_action, defender_action
                )
            except Exception as e:
                logger.warning(f"Environment error: {e}")
                done = True
                rewards = {'attacker': 0, 'defender': 0}
                info = {}
                next_state = state
            
            # Сохранение опыта
            self.attacker_memory.append({
                'state': state.copy() if isinstance(state, dict) else state,
                'action': attacker_action,
                'reward': rewards.get('attacker', 0),
                'next_state': next_state.copy() if isinstance(next_state, dict) else next_state,
                'done': done
            })
            
            self.defender_memory.append({
                'state': state.copy() if isinstance(state, dict) else state,
                'action': defender_action,
                'reward': rewards.get('defender', 0),
                'next_state': next_state.copy() if isinstance(next_state, dict) else next_state,
                'done': done
            })
            
            episode_rewards['attacker'] += rewards.get('attacker', 0)
            episode_rewards['defender'] += rewards.get('defender', 0)
            episode_steps += 1
            
            state = next_state
            
            # Обновление моделей
            if len(self.attacker_memory) >= self.config.batch_size:
                self._update_models()
        
        # Обновление статистики
        self.episode_count += 1
        self.metrics['attacker_rewards'].append(episode_rewards['attacker'])
        self.metrics['defender_rewards'].append(episode_rewards['defender'])
        self.metrics['episode_length'].append(episode_steps)
        
        # Определение победителя
        if episode_rewards['attacker'] > episode_rewards['defender']:
            self.metrics['attacker_win_rate'].append(1)
            self.metrics['defender_win_rate'].append(0)
        else:
            self.metrics['attacker_win_rate'].append(0)
            self.metrics['defender_win_rate'].append(1)
        
        # Авто-тюнинг
        if self.episode_count % 10 == 0:
            self._auto_tune_hyperparams(episode_rewards)
        
        return {
            'episode': self.episode_count,
            'attacker_reward': episode_rewards['attacker'],
            'defender_reward': episode_rewards['defender'],
            'length': episode_steps,
            'info': info
        }
    
    def _update_models(self):
        """Обновление моделей"""
        if len(self.attacker_memory) < self.config.batch_size:
            return
        
        try:
            # Обновление атакующего
            attacker_batch = random.sample(
                list(self.attacker_memory), 
                min(self.config.batch_size, len(self.attacker_memory))
            )
            attacker_loss = self._compute_loss(attacker_batch, self.attacker)
            
            if attacker_loss is not None and not torch.isnan(attacker_loss):
                self.attacker_optimizer.zero_grad()
                attacker_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.attacker.parameters(), 0.5)
                self.attacker_optimizer.step()
                self.metrics['attacker_loss'].append(attacker_loss.item())
            
            # Обновление защитника
            defender_batch = random.sample(
                list(self.defender_memory),
                min(self.config.batch_size, len(self.defender_memory))
            )
            defender_loss = self._compute_loss(defender_batch, self.defender)
            
            if defender_loss is not None and not torch.isnan(defender_loss):
                self.defender_optimizer.zero_grad()
                defender_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.defender.parameters(), 0.5)
                self.defender_optimizer.step()
                self.metrics['defender_loss'].append(defender_loss.item())
                
        except Exception as e:
            logger.warning(f"Update error: {e}")
    
    def _compute_loss(self, batch, model):
        """Вычисление loss"""
        try:
            total_loss = 0.0
            count = 0
            
            for item in batch:
                state = item.get('state', {})
                reward = item.get('reward', 0)
                
                model_input = self._prepare_model_input(state)
                output = model(model_input)
                value = output.get('value')
                
                if value is not None:
                    if isinstance(value, torch.Tensor):
                        if value.dim() == 0:
                            value = value.unsqueeze(0)
                        loss = (value - reward) ** 2
                        total_loss += loss.mean()
                        count += 1
            
            if count > 0:
                return total_loss / count
            else:
                return None
                
        except Exception as e:
            logger.warning(f"Loss computation error: {e}")
            return None
    
    def _auto_tune_hyperparams(self, episode_rewards):
        """Автоматическая настройка гиперпараметров"""
        avg_reward = (episode_rewards['attacker'] + episode_rewards['defender']) / 2
        
        try:
            new_params = self.auto_tune.suggest_params()
            
            self.config.learning_rate = new_params.get('learning_rate', self.config.learning_rate)
            self.config.batch_size = new_params.get('batch_size', self.config.batch_size)
            self.config.n_steps = new_params.get('n_steps', self.config.n_steps)
            self.config.gamma = new_params.get('gamma', self.config.gamma)
            self.config.gae_lambda = new_params.get('gae_lambda', self.config.gae_lambda)
            
            self.attacker_optimizer.param_groups[0]['lr'] = self.config.learning_rate
            self.defender_optimizer.param_groups[0]['lr'] = self.config.learning_rate
            
            self.auto_tune.update(new_params, avg_reward, {
                'attacker_reward': episode_rewards['attacker'],
                'defender_reward': episode_rewards['defender']
            })
            
            logger.info(f"Auto-tuned: LR={self.config.learning_rate:.6f}, "
                       f"Batch={self.config.batch_size}")
        except Exception as e:
            logger.warning(f"Auto-tune error: {e}")
    
    def save_models(self, path: str):
        """Сохранение моделей"""
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'attacker': self.attacker.state_dict(),
                'defender': self.defender.state_dict(),
                'config': self.config.__dict__,
                'metrics': self.metrics,
                'episode': self.episode_count
            }, f"{path}/checkpoint_{self.episode_count}.pt")
            
            torch.save(self.attacker.state_dict(), f"{path}/attacker_latest.pt")
            torch.save(self.defender.state_dict(), f"{path}/defender_latest.pt")
            
            logger.info(f"Models saved to {path}")
        except Exception as e:
            logger.error(f"Save error: {e}")
    
    def get_stats(self) -> Dict:
        """Получение статистики"""
        if len(self.metrics['attacker_rewards']) == 0:
            return {}
        
        window = min(100, len(self.metrics['attacker_rewards']))
        
        return {
            'episodes': self.episode_count,
            'attacker_reward': np.mean(self.metrics['attacker_rewards'][-window:]),
            'defender_reward': np.mean(self.metrics['defender_rewards'][-window:]),
            'attacker_win_rate': np.mean(self.metrics['attacker_win_rate'][-window:]) * 100,
            'defender_win_rate': np.mean(self.metrics['defender_win_rate'][-window:]) * 100,
            'avg_length': np.mean(self.metrics['episode_length'][-window:]),
            'attacker_loss': np.mean(self.metrics['attacker_loss'][-window:]) if self.metrics['attacker_loss'] else 0,
            'defender_loss': np.mean(self.metrics['defender_loss'][-window:]) if self.metrics['defender_loss'] else 0
        }