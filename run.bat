@echo off
setlocal
echo ========================================
echo   SecureBank - Autenticacao Facial
echo   FIAP MBA Machine Learning
echo ========================================
echo.

:: Verifica Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERRO] Python nao encontrado.
    echo Baixe em: https://www.python.org/downloads/
    pause & exit /b 1
)

:: Cria ambiente virtual se nao existir
if not exist ".venv\Scripts\activate.bat" (
    echo Criando ambiente virtual...
    python -m venv .venv
)

:: Ativa venv e instala dependencias
echo Ativando ambiente virtual...
call .venv\Scripts\activate.bat

echo Verificando dependencias...
pip install -q -r requirements.txt

echo.
echo Iniciando SecureBank...
echo.
python app.py

deactivate
endlocal
