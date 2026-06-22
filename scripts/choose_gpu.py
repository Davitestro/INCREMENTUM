# scripts/choose_gpu.py
import torch
import os

def show_gpu_info():
    """Показывает информацию о всех GPU"""
    print("=" * 60)
    print("GPU INFORMATION")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA is not available")
        return
    
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print()
    
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
        print(f"  Compute Capability: {torch.cuda.get_device_capability(i)}")
        print()

def set_gpu(gpu_id):
    """Установка конкретного GPU"""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    print(f"✅ Set CUDA_VISIBLE_DEVICES={gpu_id}")
    print(f"   Available devices: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"   Current device: {torch.cuda.get_device_name(0)}")

if __name__ == "__main__":
    show_gpu_info()
    
    # Пример использования
    print("To use GPU 3:")
    print("  python3 scripts/choose_gpu.py --use 3")
    
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == '--use':
        gpu_id = int(sys.argv[2])
        set_gpu(gpu_id)