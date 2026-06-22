# training/train_server.py
import torch
import numpy as np
import time
import json
from pathlib import Path
import logging
import sys
import os

# Добавляем путь к корню проекта
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

# Абсолютные импорты
from config.attacker_config import AttackerConfig
from config.defender_config import DefenderConfig
from models.adversarial_trainer import AdversarialTrainer
from training.metrics_logger import MetricsLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServerTrainer:
    """
    Серверное обучение с автоматической настройкой
    """
    def __init__(self, gpu_id=None):
        # Создаём папки
        Path("models_saved").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        
        # Настройка GPU
        self.gpu_id = gpu_id
        if self.gpu_id is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
            logger.info(f"Using GPU: {self.gpu_id} (CUDA_VISIBLE_DEVICES={self.gpu_id})")
        
        # Конфигурации с авто-тюнингом
        self.attacker_config = AttackerConfig()
        self.defender_config = DefenderConfig()
        
        # Тренер
        self.trainer = AdversarialTrainer(self.attacker_config)
        
        # Логгер
        self.metrics_logger = MetricsLogger()
        
        # Параметры обучения
        self.max_episodes = 10000
        self.save_interval = 100
        self.eval_interval = 50
        
        # GPU мониторинг
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            logger.info(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"Current device: {torch.cuda.current_device()}")
        else:
            logger.info("Using CPU for training")
    
    def train(self):
        """Основной цикл обучения"""
        logger.info("Starting adversarial training...")
        
        # Прогресс-бар
        pbar = tqdm(total=self.max_episodes, desc="Training")
        
        start_time = time.time()
        consecutive_errors = 0
        
        for episode in range(self.max_episodes):
            try:
                # Обучение эпизода
                results = self.trainer.train_episode()
                consecutive_errors = 0
                
                # Логирование
                if self.metrics_logger:
                    self.metrics_logger.log_episode(results)
                
                # Обновление прогресс-бара
                pbar.update(1)
                pbar.set_postfix({
                    'Att': f"{results['attacker_reward']:.1f}",
                    'Def': f"{results['defender_reward']:.1f}",
                    'Len': results['length']
                })
                
                # Оценка производительности
                if episode % self.eval_interval == 0:
                    self._evaluate_performance(episode)
                
                # Сохранение моделей
                if episode % self.save_interval == 0 and episode > 0:
                    self.trainer.save_models(f"models_saved/checkpoint_{episode}")
                    
            except KeyboardInterrupt:
                logger.info("Training interrupted by user")
                break
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Error in episode {episode}: {e}")
                
                # Если слишком много ошибок подряд - сбрасываем
                if consecutive_errors > 10:
                    logger.warning("Too many consecutive errors, resetting...")
                    self.trainer.env.reset()
                    consecutive_errors = 0
                
                continue
        
        pbar.close()
        
        # Финальное сохранение
        self.trainer.save_models("models_saved/final")
        
        # Сохранение метрик
        if self.metrics_logger:
            self.metrics_logger.save("logs/metrics_final.json")
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time/3600:.2f} hours")
    
    def _evaluate_performance(self, episode):
        """Оценка производительности"""
        if not self.metrics_logger:
            return
            
        win_rate = self.metrics_logger.get_win_rate(window=100)
        
        logger.info(f"Episode {episode}:")
        logger.info(f"  Attacker win rate: {win_rate.get('attacker', 0):.2f}%")
        logger.info(f"  Defender win rate: {win_rate.get('defender', 0):.2f}%")
        logger.info(f"  Best reward: {self.metrics_logger.get_best_reward():.2f}")
        
        # Сохранение в файл
        try:
            with open("logs/performance.json", "a") as f:
                json.dump({
                    'episode': episode,
                    'win_rate': win_rate,
                    'time': time.time()
                }, f)
                f.write("\n")
        except:
            pass

if __name__ == "__main__":
    # Используем последний GPU (cuda:3)
    trainer = ServerTrainer(gpu_id=3)
    trainer.train()