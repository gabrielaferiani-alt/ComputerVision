"""
SecureBank — Sistema de Autenticacao Facial
FIAP MBA Machine Learning & IA — Visao Computacional

Interface via OpenCV — sem Tkinter, funciona em macOS, Windows e Linux.
Fluxo: Deteccao de Face -> Vivacidade (piscadas) -> Identificacao LBPH
"""
from __future__ import annotations

import os, sys, time
import cv2
import numpy as np
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modules.detector import FaceDetector
from modules.liveness import LivenessDetector
from modules.recognizer import FaceRecognizer

# ── Janela ────────────────────────────────────────────────────────────────────
WIN  = "SecureBank - Autenticacao Facial"
W, H = 900, 560
CX, CY, CW, CH = 18, 58, 560, 420   # área da câmera no frame

FACE_CROP  = 200
N_FOTOS    = 50
CAP_INTER  = 0.25

# ── Cores BGR ─────────────────────────────────────────────────────────────────
BG    = ( 10,  22,  40)
HDR   = ( 30,  58,  95)
PANEL = ( 18,  35,  55)
BLUE  = (235,  99,  37)
GREEN = ( 16, 150, 105)
RED   = ( 50,  50, 220)
AMBER = ( 30, 140, 200)
WHITE = (252, 250, 248)
MUTED = (160, 155, 148)

SF  = cv2.FONT_HERSHEY_SIMPLEX
SFB = cv2.FONT_HERSHEY_DUPLEX


# ── Helpers de desenho ────────────────────────────────────────────────────────

def canvas() -> np.ndarray:
    f = np.zeros((H, W, 3), dtype=np.uint8)
    f[:] = BG
    return f

def draw_header(f: np.ndarray, title: str):
    cv2.rectangle(f, (0, 0), (W, 52), HDR, -1)
    cv2.putText(f, "SecureBank  |  " + title, (18, 35), SFB, 0.65, WHITE, 1)
    cv2.putText(f, "[ESC] Voltar", (W - 150, 35), SF, 0.45, MUTED, 1)

def draw_panel(f: np.ndarray):
    cv2.rectangle(f, (584, 55), (892, H - 6), PANEL, -1)

def put(f, text, x, y, color=WHITE, scale=0.55, thick=1, font=SF):
    cv2.putText(f, str(text), (x, y), font, scale, color, thick)

def draw_step(f: np.ndarray, y: int, label: str, status: str):
    cfg = {"pending": ("o", MUTED), "ok": ("V", GREEN),
           "fail": ("X", RED), "active": (">", AMBER)}
    sym, col = cfg.get(status, cfg["pending"])
    cv2.putText(f, sym,   (596, y), SFB, 0.5, col, 1)
    cv2.putText(f, label, (618, y), SF,  0.52, col, 1)

def draw_bar(f, x, y, w, h, value, total, color):
    cv2.rectangle(f, (x, y), (x + w, y + h), (40, 55, 70), -1)
    fill = int(w * min(value, total) / max(total, 1))
    if fill > 0:
        cv2.rectangle(f, (x, y), (x + fill, y + h), color, -1)
    cv2.rectangle(f, (x, y), (x + w, y + h), MUTED, 1)


# ── Aplicação ─────────────────────────────────────────────────────────────────

class App:
    def __init__(self):
        base      = os.path.dirname(os.path.abspath(__file__))
        data_dir  = os.path.join(base, "data", "faces")
        model_dir = os.path.join(base, "models")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        self.data_dir   = data_dir
        self.detector   = FaceDetector()
        self.liveness   = LivenessDetector(timeout=12, blinks_required=2)
        self.recognizer = FaceRecognizer(model_dir)
        self.cap: Optional[cv2.VideoCapture] = None

        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)

    # ── Loop principal ────────────────────────────────────────────────────────
    def run(self):
        print("SecureBank iniciado. Use a janela que abriu.")
        while True:
            action = self._menu()
            if action == "c":
                self._register()
            elif action == "a":
                self._authenticate()
            else:
                break
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    # ── Tela de menu ──────────────────────────────────────────────────────────
    def _menu(self) -> str:
        while True:
            f = canvas()
            cv2.putText(f, "SecureBank",  (W//2 - 155, 130), SFB, 1.7, BLUE, 2)
            cv2.putText(f, "Autenticacao Facial Biometrica",
                        (W//2 - 200, 172), SF, 0.65, WHITE, 1)
            cv2.line(f, (120, 198), (W - 120, 198), MUTED, 1)

            put(f, "[C]  Cadastrar Cliente",  W//2 - 155, 270, WHITE, 0.72, 1, SFB)
            put(f, "[A]  Autenticar Cliente", W//2 - 155, 330, WHITE, 0.72, 1, SFB)
            put(f, "[Q]  Sair",               W//2 - 155, 390, MUTED, 0.72, 1, SFB)

            cv2.line(f, (120, 430), (W - 120, 430), MUTED, 1)
            put(f, "Etapas: Deteccao -> Vivacidade -> Identificacao",
                W//2 - 255, 465, MUTED, 0.5)

            has = self.recognizer.has_model()
            put(f, "Clientes: cadastrados" if has else "Clientes: nenhum (cadastre primeiro)",
                W//2 - 155, 510, GREEN if has else AMBER, 0.5)

            cv2.imshow(WIN, f)
            k = cv2.waitKey(30) & 0xFF
            if k in (ord("c"), ord("C")): return "c"
            if k in (ord("a"), ord("A")): return "a"
            if k in (ord("q"), ord("Q"), 27): return "q"

    # ── Input de nome (digitação direta na janela OpenCV) ─────────────────────
    def _input_name(self) -> Optional[str]:
        buf = ""
        while True:
            f = canvas()
            bx, by = 150, 165
            cv2.rectangle(f, (bx, by), (bx + 600, by + 190), PANEL, -1)
            cv2.rectangle(f, (bx, by), (bx + 600, by + 190), BLUE, 2)

            put(f, "CADASTRO DE CLIENTE", bx + 18, by + 35, WHITE, 0.72, 1, SFB)
            cv2.line(f, (bx + 10, by + 50), (bx + 590, by + 50), MUTED, 1)
            put(f, "Nome do cliente:", bx + 18, by + 82, MUTED, 0.55)

            cv2.rectangle(f, (bx + 18, by + 95), (bx + 582, by + 140), (10, 20, 35), -1)
            cv2.rectangle(f, (bx + 18, by + 95), (bx + 582, by + 140), MUTED, 1)
            put(f, buf + "|", bx + 28, by + 130, WHITE, 0.7, 1, SFB)

            put(f, "[ENTER] Confirmar     [ESC] Cancelar",
                bx + 18, by + 175, MUTED, 0.45)

            cv2.imshow(WIN, f)
            k = cv2.waitKey(30)
            if k == -1: continue
            if k == 27: return None
            if k in (13, 10): return buf.strip() or None
            if k in (8, 127): buf = buf[:-1]
            elif 32 <= k < 128: buf += chr(k)

    # ── Cadastro ──────────────────────────────────────────────────────────────
    def _register(self):
        name = self._input_name()
        if not name:
            return

        user_id  = name.lower().replace(" ", "_")
        user_dir = os.path.join(self.data_dir, user_id)
        os.makedirs(user_dir, exist_ok=True)

        self._open_cam()
        count, last_cap = 0, 0.0
        status, s_color = "Posicione o rosto na camera", AMBER

        while count < N_FOTOS:
            ret, raw = self.cap.read()
            if not ret: continue
            raw   = cv2.flip(raw, 1)
            faces, gray = self.detector.detect_faces(raw)
            cam   = cv2.resize(raw, (CW, CH))
            sx, sy = CW / raw.shape[1], CH / raw.shape[0]

            f = canvas()
            draw_header(f, f"Cadastrando: {name}")
            f[CY:CY+CH, CX:CX+CW] = cam
            draw_panel(f)

            put(f, "PROGRESSO", 598, 88, MUTED, 0.48)
            put(f, f"{count} / {N_FOTOS} fotos", 598, 112, WHITE, 0.65, 1, SFB)
            draw_bar(f, 594, 122, 290, 14, count, N_FOTOS, BLUE)

            put(f, "INSTRUCOES", 598, 162, MUTED, 0.48)
            for i, l in enumerate(["Olhe para a camera",
                                    "Boa iluminacao",
                                    "Varie o angulo"]):
                put(f, l, 598, 185 + i * 22, WHITE, 0.5)

            put(f, "STATUS", 598, 270, MUTED, 0.48)
            put(f, status, 598, 292, s_color, 0.52)
            put(f, "[ESC] Cancelar", 598, H - 16, MUTED, 0.43)

            if len(faces) == 1:
                x, y, w, h = faces[0]
                cv2.rectangle(f,
                    (CX + int(x*sx), CY + int(y*sy)),
                    (CX + int((x+w)*sx), CY + int((y+h)*sy)), BLUE, 2)
                now = time.time()
                if now - last_cap >= CAP_INTER:
                    roi = cv2.resize(gray[y:y+h, x:x+w], (FACE_CROP, FACE_CROP))
                    cv2.imwrite(os.path.join(user_dir, f"{count:04d}.jpg"), roi)
                    count += 1
                    last_cap = now
                status, s_color = f"Capturando... {count}/{N_FOTOS}", GREEN
            elif len(faces) > 1:
                status, s_color = "Multiplos rostos! Fique sozinho", RED
            else:
                status, s_color = "Nenhum rosto detectado", AMBER

            cv2.imshow(WIN, f)
            if cv2.waitKey(30) & 0xFF == 27:
                self._close_cam()
                return

        self._close_cam()

        # Treinar modelo
        f = canvas()
        put(f, "Treinando modelo, aguarde...", W//2 - 200, H//2, AMBER, 0.8, 2, SFB)
        cv2.imshow(WIN, f)
        cv2.waitKey(1)

        self.recognizer.train(self.data_dir)

        f = canvas()
        put(f, f"Cliente '{name}' cadastrado!", W//2 - 230, H//2 - 20, GREEN, 0.8, 2, SFB)
        put(f, "Pressione qualquer tecla para voltar ao menu",
            W//2 - 250, H//2 + 30, MUTED, 0.52)
        cv2.imshow(WIN, f)
        cv2.waitKey(0)

    # ── Autenticação ──────────────────────────────────────────────────────────
    def _authenticate(self):
        if not self.recognizer.has_model():
            f = canvas()
            put(f, "Nenhum cliente cadastrado!", W//2 - 200, H//2 - 20, RED, 0.8, 2, SFB)
            put(f, "Cadastre um cliente primeiro.  [ENTER] Voltar",
                W//2 - 260, H//2 + 30, MUTED, 0.52)
            cv2.imshow(WIN, f)
            cv2.waitKey(0)
            return

        self._open_cam()
        self.liveness.reset()

        phase      = "detect"
        face_hold  = None
        steps      = {"detect": "pending", "liveness": "pending", "identity": "pending"}
        res_msg    = ""
        res_col    = WHITE
        res_sub    = ""

        while True:
            ret, raw = self.cap.read()
            if not ret: continue
            raw   = cv2.flip(raw, 1)
            faces, gray = self.detector.detect_faces(raw)
            cam   = cv2.resize(raw, (CW, CH))
            sx, sy = CW / raw.shape[1], CH / raw.shape[0]

            f = canvas()
            draw_header(f, "Autenticacao Biometrica")
            f[CY:CY+CH, CX:CX+CW] = cam
            draw_panel(f)

            put(f, "ETAPAS", 598, 85, MUTED, 0.48)
            draw_step(f, 112, "Deteccao de Face", steps["detect"])
            draw_step(f, 138, "Vivacidade",        steps["liveness"])
            draw_step(f, 164, "Identificacao",     steps["identity"])
            cv2.line(f, (588, 180), (888, 180), (40, 55, 70), 1)

            instr_lines = []

            # ── Etapa 1: detecção ─────────────────────────────────────────
            if phase == "detect":
                instr_lines = ["Posicione o rosto", "na camera"]
                if len(faces) == 1:
                    x, y, w, h = faces[0]
                    is_real = self.liveness.check_texture(gray[y:y+h, x:x+w])
                    col = BLUE if is_real else RED
                    cv2.rectangle(f,
                        (CX+int(x*sx), CY+int(y*sy)),
                        (CX+int((x+w)*sx), CY+int((y+h)*sy)), col, 2)
                    if is_real:
                        steps["detect"] = "ok"
                        face_hold = face_hold or time.time()
                        if time.time() - face_hold >= 1.5:
                            phase = "liveness"
                            self.liveness.reset()
                            face_hold = None
                    else:
                        steps["detect"] = "fail"
                        face_hold = None
                        instr_lines = ["FOTO ESTATICA detectada!", "Mostre seu rosto real"]
                else:
                    steps["detect"] = "pending"
                    face_hold = None

            # ── Etapa 2: vivacidade ───────────────────────────────────────
            elif phase == "liveness":
                rem = self.liveness.time_remaining()
                instr_lines = ["PISQUE OS OLHOS", "2x para confirmar", "vivacidade"]
                put(f, f"Tempo: {rem:.0f}s", 598, 198, AMBER, 0.52)

                if len(faces) == 1:
                    x, y, w, h = faces[0]
                    cv2.rectangle(f,
                        (CX+int(x*sx), CY+int(y*sy)),
                        (CX+int((x+w)*sx), CY+int((y+h)*sy)), AMBER, 2)
                    eyes = self.detector.detect_eyes(gray[y:y+h, x:x+w])
                    self.liveness.process_eyes(eyes)
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(f,
                            (CX+int((x+ex)*sx), CY+int((y+ey)*sy)),
                            (CX+int((x+ex+ew)*sx), CY+int((y+ey+eh)*sy)), AMBER, 1)

                blinks = self.liveness.blink_count
                put(f, f"Piscadas: {blinks} / 2", 598, 224, WHITE, 0.62, 1, SFB)
                eye_txt = "ABERTOS" if self.liveness.eye_state == "open" else "FECHADOS"
                put(f, f"Olhos: {eye_txt}", 598, 248, MUTED, 0.48)
                steps["liveness"] = "active"

                if self.liveness.is_live:
                    phase = "identity"
                    steps["liveness"] = "ok"
                    instr_lines = ["Vivacidade confirmada!"]

                if self.liveness.is_timed_out() and not self.liveness.is_live:
                    phase = "detect"
                    self.liveness.reset()
                    steps["liveness"] = "fail"
                    face_hold = None
                    instr_lines = ["Tempo esgotado!", "Tente novamente"]

            # ── Etapa 3: identificação ────────────────────────────────────
            elif phase == "identity":
                instr_lines = ["Identificando..."]
                if len(faces) == 1:
                    x, y, w, h = faces[0]
                    roi = cv2.resize(gray[y:y+h, x:x+w], (FACE_CROP, FACE_CROP))
                    name, conf = self.recognizer.predict(roi)
                    if conf < FaceRecognizer.CONFIDENCE_THRESHOLD:
                        cv2.rectangle(f,
                            (CX+int(x*sx), CY+int(y*sy)),
                            (CX+int((x+w)*sx), CY+int((y+h)*sy)), GREEN, 3)
                        steps["identity"] = "ok"
                        phase   = "done"
                        res_msg = "ACESSO LIBERADO"
                        res_col = GREEN
                        res_sub = f"Cliente: {name}   Confianca: {100-conf:.0f}%"
                    elif name:
                        cv2.rectangle(f,
                            (CX+int(x*sx), CY+int(y*sy)),
                            (CX+int((x+w)*sx), CY+int((y+h)*sy)), RED, 3)
                        steps["identity"] = "fail"
                        phase   = "done"
                        res_msg = "ACESSO NEGADO"
                        res_col = RED
                        res_sub = "Nao identificado. Log enviado a IA."

            # ── Concluído ─────────────────────────────────────────────────
            elif phase == "done":
                if len(faces) == 1:
                    c = GREEN if res_col == GREEN else RED
                    x, y, w, h = faces[0]
                    cv2.rectangle(f,
                        (CX+int(x*sx), CY+int(y*sy)),
                        (CX+int((x+w)*sx), CY+int((y+h)*sy)), c, 3)

            # ── Instrução no painel ───────────────────────────────────────
            iy = 295
            for line in instr_lines:
                put(f, line, 598, iy, AMBER if phase != "done" else WHITE, 0.52)
                iy += 24

            # ── Caixa de resultado ────────────────────────────────────────
            if res_msg:
                cv2.line(f, (588, 385), (888, 385), (40, 55, 70), 1)
                bg_res = (12, 55, 12) if res_col == GREEN else (12, 12, 55)
                cv2.rectangle(f, (590, 393), (886, 498), bg_res, -1)
                cv2.rectangle(f, (590, 393), (886, 498), res_col, 2)
                put(f, res_msg, 606, 432, res_col, 0.75, 2, SFB)
                put(f, res_sub, 606, 460, WHITE, 0.42)
                put(f, "[ENTER] Nova verificacao", 606, 490, MUTED, 0.42)

            put(f, "[ESC] Menu principal", 598, H - 14, MUTED, 0.42)
            put(f, "SecureBank | Biometria", CX + 8, CY + 16, MUTED, 0.4)

            cv2.imshow(WIN, f)
            k = cv2.waitKey(30) & 0xFF

            if k == 27:
                break
            if k in (13, 10) and phase == "done":
                phase = "detect"
                face_hold = None
                res_msg = res_sub = ""
                steps = {"detect":"pending","liveness":"pending","identity":"pending"}
                self.liveness.reset()

        self._close_cam()

    # ── Camera ────────────────────────────────────────────────────────────────
    def _open_cam(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def _close_cam(self):
        if self.cap:
            self.cap.release()
            self.cap = None


if __name__ == "__main__":
    print("Iniciando SecureBank...")
    App().run()
    print("Encerrado.")
