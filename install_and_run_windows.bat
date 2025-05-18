@echo off
echo ===================================================
echo  Object Detection Application - Setup and Run
echo ===================================================
echo.

REM Check if Python is installed
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in your PATH.
    echo Please install Python 3.8 or newer from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

echo Python is installed. Installing dependencies...

REM Create a virtual environment (optional)
echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate

REM Install dependencies
echo Installing required packages...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ===================================================
echo  Setup complete! Starting the application...
echo ===================================================

REM Run the Flask application
python -m flask run --debug

pause 