# run_train.py
import os
import sys
import subprocess

def run_on_gpu(gpu_id=3):
    """Запуск обучения на указанном GPU"""
    
    # Устанавливаем переменную окружения
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    print("=" * 60)
    print(f"🚀 Training on GPU {gpu_id}")
    print(f"   CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}")
    print("=" * 60)
    
    # Импортируем и запускаем обучение
    from training.train_server import ServerTrainer
    trainer = ServerTrainer()
    trainer.max_episodes = 1000
    trainer.train()

if __name__ == "__main__":
    # По умолчанию используем GPU 3
    gpu_id = 3
    if len(sys.argv) > 1:
        gpu_id = int(sys.argv[1])
    
    run_on_gpu(gpu_id)