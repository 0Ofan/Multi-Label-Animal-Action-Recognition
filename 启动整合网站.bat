@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
 echo Create .venv and install requirements.txt first. See delivery instructions.
 pause
 exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8504 --global.developmentMode false
pause
