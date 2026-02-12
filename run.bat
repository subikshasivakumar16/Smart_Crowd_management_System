@echo off
cd /d "%~dp0"
if not exist "models\lstm_model.keras" python setup_models.py
python app.py
pause
