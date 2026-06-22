#!/bin/bash
# run_train.sh - Скрипт для запуска обучения на конкретном GPU

# Очищаем переменную, чтобы выбрать конкретный GPU
unset CUDA_VISIBLE_DEVICES

# Используем GPU 3 (последний)
export CUDA_VISIBLE_DEVICES=3

echo "======================================"
echo "Starting training on GPU 3"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "======================================"

# Запуск обучения
python3 main.py --mode train --episodes 1000

# Или напрямую
# python3 training/train_server.py