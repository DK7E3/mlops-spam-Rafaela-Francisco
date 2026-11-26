"""Testes funcionais e de contrato dos endpoints da API (/health, /predict, /metrics e /)."""
import os

from fastapi.testclient import TestClient

# Garante que durante os testes as variáveis de ambiente sejam definidas ANTES de importar a app
os.environ["API_KEY"] = "chave-ficticia-123"
os.environ["MODEL_VERSION"] = "v1"
os.environ["APP_ENV"] = "test"

from app.main import app

client = TestClient(app)


def test_health_check():
    """O endpoint /health deve retornar 200 e dados de status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["versao"] == "v1"
    assert data["ambiente"] in ["dev", "test", "prod"]
    assert "X-Request-ID" in response.headers


def test_ready_check():
    """O endpoint /ready (readiness probe) deve retornar 200 quando o modelo está carregado."""
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "pronto"
    assert data["versao"] == "v1"


def test_predict_sucesso():
    """Chamada com chave correta e texto suspeito deve retornar rótulo spam e probabilidade."""
    payload = {"texto": "Parabéns! Você ganhou um prêmio, clique no link agora"}
    headers = {"X-API-Key": "chave-ficticia-123"}
    response = client.post("/predict", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["rotulo"] in ["spam", "nao_spam"]
    assert "probabilidade" in data
    assert data["versao"] == "v1"


def test_predict_nao_spam():
    """Classificação de mensagem legítima deve retornar rótulo 'nao_spam'."""
    payload = {"texto": "A reunião foi remarcada para as 15h na sala 2"}
    headers = {"X-API-Key": "chave-ficticia-123"}
    response = client.post("/predict", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["rotulo"] == "nao_spam"
    assert data["probabilidade"] >= 0.5
    assert data["versao"] == "v1"


def test_metrics():
    """O endpoint /metrics deve retornar 200 e as chaves de telemetria."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "requisicoes" in data
    assert "erros_5xx" in data
    assert "latencia_media_ms" in data


def test_serve_index():
    """A rota raiz (/) deve carregar a interface web estática com sucesso."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Classificador de spam" in response.text
