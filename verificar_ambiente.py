"""
Verifica se o ambiente esta pronto para rodar o SecureBank.
Execute antes do app.py se tiver duvidas:  python verificar_ambiente.py
"""
import sys

OK = "\033[92m[OK]\033[0m"
FAIL = "\033[91m[ERRO]\033[0m"
WARN = "\033[93m[AVISO]\033[0m"

erros = 0


def check(label, fn):
    global erros
    try:
        result = fn()
        print(f"  {OK} {label}" + (f": {result}" if result else ""))
    except Exception as e:
        print(f"  {FAIL} {label}: {e}")
        erros += 1


print("\n=== Verificação de Ambiente — SecureBank ===\n")

# Python
v = sys.version_info
print(f"  {'OK' if v >= (3, 8) else 'ERRO'} Python {v.major}.{v.minor}.{v.micro}", end="")
if v < (3, 8):
    print(f" — Python 3.8+ é necessário")
    erros += 1
else:
    print()

# Pacotes
check("numpy", lambda: __import__("numpy").__version__)
check("Pillow (PIL)", lambda: __import__("PIL").__version__)

def check_opencv():
    import cv2
    v = cv2.__version__
    if not hasattr(cv2, "face"):
        raise ImportError(
            "cv2.face não encontrado — instale opencv-contrib-python, "
            "NÃO opencv-python"
        )
    return v

check("opencv-contrib-python (cv2.face)", check_opencv)

# Tkinter
def check_tk():
    import tkinter
    root = tkinter.Tk()
    root.withdraw()
    root.destroy()
    return "ok"

check("tkinter", check_tk)

# Arquivos de classificadores
import os
clf_dir = os.path.join(os.path.dirname(__file__), "classifiers")
for xml in ["haarcascade_frontalface_default.xml", "haarcascade_eye.xml"]:
    path = os.path.join(clf_dir, xml)
    if os.path.exists(path):
        print(f"  {OK} Classificador: {xml}")
    else:
        print(f"  {FAIL} Classificador ausente: {xml}")
        erros += 1

# Câmera
def check_camera():
    import cv2
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Nenhuma câmera detectada no índice 0")
    cap.release()
    return "detectada"

check("Câmera (índice 0)", check_camera)

# Resultado
print()
if erros == 0:
    print(">>> Tudo OK! Execute:  python app.py\n")
else:
    print(f"!!! {erros} problema(s) encontrado(s). Corrija antes de rodar o app.\n")
    sys.exit(1)
