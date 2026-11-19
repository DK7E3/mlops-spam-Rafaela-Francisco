import os
import time
from pathlib import Path
from typing import List

import joblib
from fastapi import FastAPI, HTTPException, Header, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# --- Carrega variáveis de ambiente de .env se existir localmente ---
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

# --- Configurações obtidas do ambiente ---
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1").strip()
APP_ENV = os.getenv("APP_ENV", "dev").strip()
API_KEY = os.getenv("API_KEY")
if API_KEY:
    API_KEY = API_KEY.strip().strip('"').strip("'")
MODEL_PATH = Path(
    os.getenv("MODEL_PATH", str(Path(__file__).resolve().parent.parent / "model" / "modelo.joblib"))
)

# --- Carrega o modelo de Machine Learning ---
modelo = joblib.load(MODEL_PATH)

app = FastAPI(
    title="API de classificação de spam",
    version=MODEL_VERSION,
    description="API de inferência para detecção de spam desenvolvida para a disciplina MLOps e Deploy."
)

# --- Métricas em memória para o Painel de Operação ---
metrics = {
    "total_requests": 0,
    "errors_5xx": 0,
    "latencies_ms": []
}

# --- Middleware de Métricas e Latência ---
@app.middleware("http")
async def track_metrics(request: Request, call_next):
    start_time = time.perf_counter()
    try:
        response: Response = await call_next(request)
        duration_ms = (time.perf_counter() - start_time) * 1000
        # Apenas rastreia endpoints da API (ignora estáticos se aplicável)
        if request.url.path in ["/predict", "/health"]:
            metrics["total_requests"] += 1
            metrics["latencies_ms"].append(duration_ms)
            if len(metrics["latencies_ms"]) > 200:
                metrics["latencies_ms"].pop(0)
            if response.status_code >= 500:
                metrics["errors_5xx"] += 1
        return response
    except Exception:
        metrics["total_requests"] += 1
        metrics["errors_5xx"] += 1
        raise

# --- Configuração de CORS (permite chamadas da página web) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Mensagem(BaseModel):
    texto: str = Field(min_length=1, max_length=2000, description="Texto a ser classificado")


@app.get("/health")
def health():
    """Endpoint de checagem de saúde (liveness probe). Não exige autenticação."""
    return {
        "status": "ok",
        "versao": MODEL_VERSION,
        "ambiente": APP_ENV
    }


@app.get("/metrics")
def get_metrics():
    """Retorna métricas operacionais para o painel."""
    latencies = metrics["latencies_ms"]
    avg_latency = round(sum(latencies) / len(latencies), 1) if latencies else 0.0
    return {
        "requisicoes": metrics["total_requests"],
        "erros_5xx": metrics["errors_5xx"],
        "latencia_media_ms": avg_latency
    }


@app.post("/predict")
def predict(msg: Mensagem, x_api_key: str | None = Header(None, alias="X-API-Key")):
    """Classifica mensagem em 'spam' ou 'nao_spam' com probabilidade."""
    # 1. Valida texto vazio ou apenas espaços
    if not msg.texto.strip():
        raise HTTPException(
            status_code=400,
            detail="O texto da mensagem não pode ser vazio ou conter apenas espaços em branco."
        )

    # 2. Valida chave de API se configurada no ambiente
    if API_KEY:
        header_key = (x_api_key or "").strip().strip('"').strip("'")
        if header_key != API_KEY:
            raise HTTPException(
                status_code=401,
                detail="Chave de API inválida ou ausente. Forneça o cabeçalho X-API-Key."
            )

    # 3. Inferência com o modelo treinado
    probabilidades = modelo.predict_proba([msg.texto])[0]
    indice = int(probabilidades.argmax())
    rotulo = str(modelo.classes_[indice])
    probabilidade = round(float(probabilidades[indice]), 2)

    return {
        "rotulo": rotulo,
        "probabilidade": probabilidade,
        "versao": MODEL_VERSION
    }


# --- Servir a Página Web Estática ---
static_dir = Path(__file__).resolve().parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    def serve_index():
        return FileResponse(str(static_dir / "index.html"))
