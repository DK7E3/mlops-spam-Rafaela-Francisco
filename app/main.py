"""API de inferência para classificação de mensagens spam (Etapa 1 - MLOps e Deploy)."""
import logging
import os
import time
import unicodedata
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException, Header, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# --- Configuração de Logging Estruturado ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("spam_api")

# --- Carrega variáveis de ambiente de .env se existir localmente ---
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

def _normalize_key(val: str | None) -> str:
    """Normaliza a chave removendo espaços, aspas, acentos e diferenças de maiúsculas/minúsculas."""
    if not val:
        return ""
    cleaned = val.strip().strip('"').strip("'").strip()
    if cleaned.lower().startswith("bearer "):
        cleaned = cleaned[7:].strip()
    cleaned = "".join(
        c for c in unicodedata.normalize("NFKD", cleaned)
        if not unicodedata.combining(c) and c.isprintable()
    )
    return cleaned.strip().lower()


# --- Configurações obtidas do ambiente ---
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1").strip()
APP_ENV = os.getenv("APP_ENV", "dev").strip()
API_KEY = _normalize_key(os.getenv("API_KEY")) or None
MODEL_PATH = Path(
    os.getenv("MODEL_PATH", str(Path(__file__).resolve().parent.parent / "model" / "modelo.joblib"))
)

# --- Carrega o modelo de Machine Learning ---
modelo = joblib.load(MODEL_PATH)
logger.info("Modelo de Machine Learning carregado com sucesso a partir de %s", MODEL_PATH)

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
    allow_credentials=False,
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
def predict(
    msg: Mensagem,
    request: Request,
    x_api_key: str | None = Header(None, alias="X-API-Key"),
):
    """Classifica mensagem em 'spam' ou 'nao_spam' com probabilidade."""
    # 1. Valida texto vazio ou apenas espaços
    if not msg.texto.strip():
        logger.warning("Tentativa de inferência rejeitada: texto vazio ou composto apenas por espaços em branco.")
        raise HTTPException(
            status_code=400,
            detail="Mensagem inválida."
        )

    # 2. Valida chave de API se configurada no ambiente
    expected_key = _normalize_key(os.getenv("API_KEY")) or API_KEY
    if expected_key:
        raw_header = x_api_key or request.headers.get("x-api-key") or request.headers.get("authorization")
        header_key = _normalize_key(raw_header)
        if not header_key or header_key != expected_key:
            logger.warning("Acesso não autorizado ao endpoint /predict: chave de API inválida ou ausente.")
            raise HTTPException(
                status_code=401,
                detail="Não autorizado."
            )

    # 3. Inferência com o modelo treinado
    probabilidades = modelo.predict_proba([msg.texto])[0]
    indice = int(probabilidades.argmax())
    rotulo = str(modelo.classes_[indice])
    probabilidade = round(float(probabilidades[indice]), 2)

    logger.info(
        "Predição realizada | rotulo=%s | probabilidade=%.2f | versao=%s",
        rotulo,
        probabilidade,
        MODEL_VERSION
    )

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
