"""API de inferência para classificação de mensagens spam (MLOps e Deploy)."""
import json
import os
import time
import unicodedata
import uuid
from pathlib import Path

import joblib
from fastapi import FastAPI, Header, HTTPException, Request, Response
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


def _normalize_key(val: str | None) -> str:
    """Normaliza a chave removendo espaços, aspas, acentos e diferenças de caixa."""
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


def log_evento(nivel: str, mensagem: str, **campos) -> None:
    """Escreve um registro de log estruturado em JSON na saída padrão (stdout)."""
    registro = {
        "hora": time.strftime("%H:%M:%S"),
        "nivel": nivel,
        "mensagem": mensagem,
        "versao": MODEL_VERSION,
        **campos,
    }
    print(json.dumps(registro, ensure_ascii=False), flush=True)


# --- Carrega o modelo de Machine Learning ---
modelo = joblib.load(MODEL_PATH)
log_evento("INFO", "serviço iniciado e modelo carregado", ambiente=APP_ENV)

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


# --- Middleware de Métricas, Latência e Trace ID ---
@app.middleware("http")
async def track_metrics(request: Request, call_next):
    """Mede latência, contabiliza requisições/erros 5xx e associa um ID por requisição."""
    id_req = str(uuid.uuid4())[:8]
    request.state.id_req = id_req
    start_time = time.perf_counter()
    try:
        response: Response = await call_next(request)
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        response.headers["X-Request-ID"] = id_req
        if request.url.path in ["/predict", "/health", "/ready"]:
            metrics["total_requests"] += 1
            metrics["latencies_ms"].append(duration_ms)
            if len(metrics["latencies_ms"]) > 200:
                metrics["latencies_ms"].pop(0)
            if response.status_code >= 500:
                metrics["errors_5xx"] += 1
                log_evento(
                    "ERROR",
                    "falha interna na requisição",
                    id=id_req,
                    rota=request.url.path,
                    status=response.status_code,
                    ms=duration_ms,
                )
        return response
    except Exception:
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        metrics["total_requests"] += 1
        metrics["errors_5xx"] += 1
        log_evento(
            "ERROR",
            "exceção não tratada na requisição",
            id=id_req,
            rota=request.url.path,
            status=500,
            ms=duration_ms,
        )
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


@app.get("/ready")
def ready():
    """Endpoint de prontidão (readiness probe). Verifica se o modelo está carregado."""
    if modelo is None:
        raise HTTPException(status_code=503, detail="Serviço indisponível.")
    return {
        "status": "pronto",
        "versao": MODEL_VERSION
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
    t0 = time.perf_counter()
    id_req = getattr(request.state, "id_req", str(uuid.uuid4())[:8])

    # 1. Valida texto vazio ou apenas espaços
    if not msg.texto.strip():
        ms = round((time.perf_counter() - t0) * 1000, 2)
        log_evento(
            "WARNING",
            "texto vazio rejeitado",
            id=id_req,
            rota="/predict",
            status=400,
            ms=ms,
        )
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
            ms = round((time.perf_counter() - t0) * 1000, 2)
            log_evento(
                "WARNING",
                "acesso não autorizado",
                id=id_req,
                rota="/predict",
                status=401,
                ms=ms,
            )
            raise HTTPException(
                status_code=401,
                detail="Não autorizado."
            )

    # 3. Inferência com o modelo treinado (sem registrar o texto do usuário nos logs)
    probabilidades = modelo.predict_proba([msg.texto])[0]
    indice = int(probabilidades.argmax())
    rotulo = str(modelo.classes_[indice])
    probabilidade = round(float(probabilidades[indice]), 2)
    ms = round((time.perf_counter() - t0) * 1000, 2)

    log_evento(
        "INFO",
        "requisição atendida",
        id=id_req,
        rota="/predict",
        status=200,
        ms=ms,
        rotulo=rotulo,
        probabilidade=probabilidade,
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
