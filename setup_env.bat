@echo off
python -m venv venv
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo Environment ready. Activate with: venv\Scripts\activate
pause
