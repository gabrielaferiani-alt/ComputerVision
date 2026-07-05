#!/usr/bin/env bash
set -e

echo "========================================"
echo "  SecureBank - Autenticacao Facial"
echo "  FIAP MBA Machine Learning"
echo "========================================"
echo

# Verifica Python 3
if ! command -v python3 &>/dev/null; then
    echo "[ERRO] python3 nao encontrado. Instale Python 3.8+."
    exit 1
fi

# Verifica tkinter (Linux pode precisar instalar separado)
if ! python3 -c "import tkinter" 2>/dev/null; then
    echo "[ERRO] tkinter nao encontrado."
    echo "Ubuntu/Debian: sudo apt install python3-tk"
    echo "Fedora:        sudo dnf install python3-tkinter"
    echo "macOS:         instale Python oficial de python.org (ja inclui tkinter)"
    exit 1
fi

# Cria venv se nao existir
if [ ! -f ".venv/bin/activate" ]; then
    echo "Criando ambiente virtual..."
    python3 -m venv .venv
fi

# Ativa e instala
source .venv/bin/activate
echo "Verificando dependencias..."
pip install -q -r requirements.txt

echo
echo "Iniciando SecureBank..."
echo
export TK_SILENCE_DEPRECATION=1
python app.py

deactivate
