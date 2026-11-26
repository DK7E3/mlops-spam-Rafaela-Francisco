"""Testes de segurança, autenticação e validação de entrada da API."""
import os

from fastapi.testclient import TestClient

os.environ["API_KEY"] = "chave-ficticia-123"
os.environ["MODEL_VERSION"] = "v1"
os.environ["APP_ENV"] = "test"

from app.main import app

client = TestClient(app)


def test_predict_sem_chave_retorna_401():
    """Se API_KEY estiver configurada, chamada sem X-API-Key deve retornar 401 com erro genérico."""
    payload = {"texto": "Você ganhou um prêmio exclusivo"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 401
    assert response.json()["detail"] == "Não autorizado."


def test_predict_chave_invalida_retorna_401():
    """Chave incorreta deve retornar 401 sem expor detalhes internos."""
    payload = {"texto": "Você ganhou um prêmio exclusivo"}
    headers = {"X-API-Key": "chave-errada-xyz"}
    response = client.post("/predict", json=payload, headers=headers)
    assert response.status_code == 401
    assert response.json()["detail"] == "Não autorizado."


def test_predict_texto_vazio_retorna_400():
    """Texto vazio ou apenas com espaços deve ser rejeitado com 400."""
    payload = {"texto": "   "}
    headers = {"X-API-Key": "chave-ficticia-123"}
    response = client.post("/predict", json=payload, headers=headers)
    assert response.status_code == 400
    assert response.json()["detail"] == "Mensagem inválida."


def test_cabecalhos_seguranca_owasp_presentes():
    """As respostas devem incluir cabeçalhos HTTP de proteção OWASP."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options") == "DENY"
    assert response.headers.get("Referrer-Policy") == "no-referrer"
