# main.py
import sys
import os
import argparse
from pathlib import Path
import torch

# Добавляем текущую директорию в PATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description='Strategic RL Training System')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'visualize', 'test'],
                       help='Режим работы')
    parser.add_argument('--model', type=str, default=None,
                       help='Путь к модели для визуализации')
    parser.add_argument('--episodes', type=int, default=10000,
                       help='Количество эпизодов для обучения')
    parser.add_argument('--gpu', type=int, default=None,
                       help='Номер GPU для использования (0-3)')
    
    args = parser.parse_args()
    
    # Настройка GPU
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        print(f"🔧 Using GPU: {args.gpu}")
        print(f"   CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    
    # Показываем информацию о GPU
    if torch.cuda.is_available():
        print(f"✅ CUDA available")
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️ CUDA not available, using CPU")
    
    if args.mode == 'train':
        print("🚀 Starting training mode...")
        from training.train_server import ServerTrainer
        trainer = ServerTrainer()
        trainer.max_episodes = args.episodes
        trainer.train()
    
    elif args.mode == 'visualize':
        print("🎮 Starting visualization mode...")
        from visualization.visualizer_laptop import LaptopVisualizer
        visualizer = LaptopVisualizer(args.model)
        visualizer.run()
    
    elif args.mode == 'test':
        print("🧪 Starting test mode...")
        from tests.test_environment import test_environment
        test_environment()

if __name__ == "__main__":
    main()