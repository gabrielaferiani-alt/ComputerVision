# SecureBank — Sistema de Autenticação Facial

**FIAP MBA em Machine Learning & Inteligência Artificial**  
**Disciplina:** Visão Computacional  
**Aluna:** Gabriela Feriani Leoni

---

## Contexto do Trabalho

O setor de fraudes identificou clientes que alegam não ter contratado serviços como crédito pessoal, mesmo que a autenticação por senha tenha sido realizada corretamente. Para mitigar esse risco, este sistema implementa uma **segunda camada de autenticação biométrica** baseada em reconhecimento facial com detecção de vivacidade.

---

## Funcionalidades

| Componente | Técnica | Detalhe |
|---|---|---|
| **Detecção de Faces** | Haar Cascade (`haarcascade_frontalface_default.xml`) | Localiza rostos em tempo real via webcam |
| **Identificação** | LBPH Face Recognizer (OpenCV) | Compara face capturada com banco de clientes cadastrados |
| **Vivacidade (Liveness)** | Detecção de piscadas + Análise de textura | Impede uso de foto estática por fraudador |

### Fluxo de Autenticação em 3 Etapas

```
Etapa 1 — Detecção de Face
  ↓ Haar Cascade + análise de textura (Laplacian variance)
  ↓ Foto estática → BLOQUEADO, evidência encaminhada para IA

Etapa 2 — Vivacidade / Liveness
  ↓ Cliente deve piscar 2× em até 12 segundos
  ↓ Piscada detectada via haarcascade_eye (open→closed→open)
  ↓ Timeout sem piscadas → reinicia verificação

Etapa 3 — Identificação
  ↓ LBPH Face Recognizer compara com banco de faces
  ↓ Confiança < 85 → ACESSO LIBERADO
  ↓ Não reconhecido → ACESSO NEGADO + log para IA
```

---

## Instalação

### Requisitos
- Python 3.10+
- Webcam
- Windows / macOS / Linux

### Passos

```bash
# 1. Entrar na pasta do projeto
cd banco-autenticacao-facial

# 2. (Opcional) Criar ambiente virtual
python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate  # macOS/Linux

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Executar
python app.py
```

**Windows — atalho:** clique duplo em `run.bat`

> ⚠️ Use **`opencv-contrib-python`** e **não** `opencv-python`.  
> Os dois pacotes conflitam. Desinstale o `opencv-python` se já tiver instalado.

---

## Uso

### Cadastrar um Cliente
1. Clique em **Cadastrar Cliente**
2. Digite o nome completo
3. Posicione o rosto diante da câmera
4. O sistema captura **50 fotos** automaticamente (a cada 0,25 s)
5. Ao final, o modelo LBPH é treinado e salvo em `models/`

### Autenticar um Cliente
1. Clique em **Autenticar Cliente**
2. **Etapa 1 (Detecção):** coloque o rosto na câmera — o sistema verifica se é real
3. **Etapa 2 (Vivacidade):** pisque os olhos 2 vezes no prazo de 12 segundos
4. **Etapa 3 (Identificação):** LBPH compara com o banco de cadastros
5. Resultado: `✓ ACESSO LIBERADO` ou `✗ ACESSO NEGADO`

---

## Arquitetura

```
banco-autenticacao-facial/
├── app.py                   # Interface Tkinter + lógica de fluxo
├── modules/
│   ├── detector.py          # FaceDetector (Haar Cascade)
│   ├── liveness.py          # LivenessDetector (piscadas + textura)
│   └── recognizer.py        # FaceRecognizer (LBPH)
├── data/
│   └── faces/
│       └── <nome_cliente>/  # 50 imagens 200×200px por cliente
├── models/
│   ├── face_model.yml       # Modelo LBPH treinado
│   └── labels.json          # Mapeamento ID → nome
├── requirements.txt
└── run.bat                  # Atalho Windows
```

---

## Links

- Youtube:https://www.youtube.com/watch?v=R8jIxPi6YKw
- GitHub:https://github.com/gabrielaferiani-alt/ComputerVision
