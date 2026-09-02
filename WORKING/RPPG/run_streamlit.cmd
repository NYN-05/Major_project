@echo off
REM run_streamlit.cmd - launch the rPPG Streamlit demo
cd /d "C:\Users\JHASHANK\Desktop\Maj_Proj\WORKING\RPPG"
set "PYTHONPATH=%PYTHONPATH%;C:\Users\JHASHANK\Desktop\Maj_Proj\WORKING\RPPG"
"C:\Users\JHASHANK\Desktop\Maj_Proj\venv\Scripts\streamlit.exe" run rppg-pipeline\streamlit_app.py %*