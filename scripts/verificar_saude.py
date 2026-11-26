"""Script de verificação de saúde, alertas operacionais e reinício controlado (Etapa 4).

Exemplos de uso:
  Verificação única (teste rápido):  python scripts/verificar_saude.py --once
  Monitoramento contínuo (a cada 10s): python scripts/verificar_saude.py
"""
import argparse
import json
import os
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

BASE_URL = os.getenv("SERVICE_URL", "http://localhost:8000").rstrip("/")
DEPLOYMENT_NAME = os.getenv("K8S_DEPLOYMENT", "deployment/spam-api")
MAX_FALHAS_SEGUIDAS = 3
LIMITE_ERROS_5XX = 5
LIMITE_LATENCIA_MS = 500.0


def log_automacao(nivel: str, mensagem: str, **campos) -> None:
    """Registra eventos e alertas da automação em formato JSON na saída padrão."""
    registro = {
        "hora": time.strftime("%H:%M:%S"),
        "nivel": nivel,
        "origem": "automacao-saude",
        "mensagem": mensagem,
        **campos,
    }
    print(json.dumps(registro, ensure_ascii=False), flush=True)


def checar_servico(url_base: str) -> tuple[bool, dict]:
    """Consulta /health e /metrics retornando (saudavel, metricas)."""
    parsed = urllib.parse.urlparse(url_base)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Esquema de URL não permitido: {parsed.scheme}")

    try:
        with urllib.request.urlopen(f"{url_base}/health", timeout=5) as resp_health:
            ok = resp_health.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False, {}

    metricas = {}
    try:
        with urllib.request.urlopen(f"{url_base}/metrics", timeout=5) as resp_metrics:
            if resp_metrics.status == 200:
                metricas = json.loads(resp_metrics.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as err:
        log_automacao("WARNING", "falha ao ler /metrics", detalhe=str(err))

    return ok, metricas


def avaliar_alertas(metricas: dict) -> None:
    """Avalia limites de erros 5xx e latência média definidos na Etapa 4."""
    erros_5xx = metricas.get("erros_5xx", 0)
    latencia_ms = metricas.get("latencia_media_ms", 0.0)

    if erros_5xx > LIMITE_ERROS_5XX:
        log_automacao(
            "ALERTA",
            "muitos erros 5xx detectados",
            erros_5xx=erros_5xx,
            limite=LIMITE_ERROS_5XX,
        )

    if latencia_ms > LIMITE_LATENCIA_MS:
        log_automacao(
            "ALERTA",
            "latência média acima do limite",
            latencia_media_ms=latencia_ms,
            limite_ms=LIMITE_LATENCIA_MS,
        )


def reiniciar_deployment() -> None:
    """Executa reinício controlado no Kubernetes após falhas consecutivas."""
    log_automacao(
        "ALERTA",
        "serviço fora do ar por 3 verificações seguidas; acionando rollout restart",
        alvo=DEPLOYMENT_NAME,
    )
    subprocess.run(
        ["kubectl", "rollout", "restart", DEPLOYMENT_NAME],
        check=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Verificador de saúde e alertas da API de spam.")
    parser.add_argument("--url", default=BASE_URL, help="URL base do serviço (padrão: http://localhost:8000)")
    parser.add_argument("--once", action="store_true", help="Executa apenas uma verificação e encerra.")
    parser.add_argument("--interval", type=int, default=10, help="Intervalo em segundos entre verificações.")
    args = parser.parse_args()

    falhas = 0
    while True:
        ok, metricas = checar_servico(args.url.rstrip("/"))
        falhas = 0 if ok else falhas + 1

        if ok:
            log_automacao(
                "INFO",
                "OK: serviço saudável",
                url=args.url,
                requisicoes=metricas.get("requisicoes", 0),
                erros_5xx=metricas.get("erros_5xx", 0),
                latencia_media_ms=metricas.get("latencia_media_ms", 0.0),
            )
            avaliar_alertas(metricas)
        else:
            log_automacao(
                "WARNING",
                f"ALERTA: serviço fora do ar (falha {falhas}/{MAX_FALHAS_SEGUIDAS})",
                url=args.url,
                falhas_seguidas=falhas,
            )
            if falhas >= MAX_FALHAS_SEGUIDAS:
                reiniciar_deployment()
                falhas = 0

        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
