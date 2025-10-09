@echo off
call venv\Scripts\activate
python src/task1/train_mnist.py --algo cnn --epochs 3
pause
