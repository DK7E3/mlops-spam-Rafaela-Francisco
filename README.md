# 🛡️ Classificador de Spam — Do Notebook à Produção (Etapa 1)

> **Componente Curricular:** MLOps e Deploy  
> **Dupla:** Rafaela & Francisco  
> **Repositório:** [https://github.com/DK7E3/mlops-spam-Rafaela-Francisco](https://github.com/DK7E3/mlops-spam-Rafaela-Francisco)  
> **Versão de Entrega:** `v1`  
> **URL da Aplicação Online (Render):** `https://mlops-spam-rafaela-francisco.onrender.com` *(em processo de publicação)*

---

## 1. Visão Geral do Projeto

Este projeto implementa a infraestrutura de entrega e operação em produção (MLOps) para um modelo de classificação de mensagens de texto em **spam** ou **não spam**. 

O modelo pré-treinado (`model/modelo.joblib`), baseado em **TF-IDF + Regressão Logística**, foi desacoplado do ambiente exploratório (notebooks) e integrado a uma **API REST de alta performance** com interface visual interativa e painel de métricas operacionais em tempo real.

```
       ┌────────────────────────┐
       │   Página Web (UI)      │
       │ (Dark Mode / Métricas) │
       └───────────┬────────────┘
                   │ HTTP / Fetch (com X-API-Key)
                   ▼
       ┌────────────────────────┐
       │    API FastAPI         │
       │  /health  /predict     │
       └───────────┬────────────┘
                   │ In-Memory Inference
                   ▼
       ┌────────────────────────┐
       │  Modelo scikit-learn   │
       │    (modelo.joblib)     │
       └────────────────────────┘
```

---

## 2. Tecnologias Escolhidas e Justificativas Técnicas

| Componente | Tecnologia | Justificativa Técnica |
|---|---|---|
| **Linguagem** | Python 3.12 | Padrão da indústria para Ciência de Dados e MLOps, oferecendo ampla compatibilidade e ótimo desempenho de I/O assíncrono. |
| **Framework API** | **FastAPI** (`0.115.6`) | Escolhido pela alta performance com ASGI (`async`), validação de dados automática e estrita via Pydantic, geração nativa de documentação Swagger (`/docs`) e baixo overhead. |
| **Servidor ASGI** | **Uvicorn** (`0.34.0`) | Servidor rápido e padrão de mercado para execução de aplicações FastAPI. |
| **Machine Learning** | **scikit-learn** (`1.8.0`) + **Joblib** (`1.6.0`) | Versão obrigatória e exata para carregamento íntegro do artefato serializado `modelo.joblib`, evitando incompatibilidades de desserialização (`InconsistentVersionWarning`). |
| **Interface Web** | HTML5, CSS3 Moderno e Vanilla JavaScript | Desenvolvido sem frameworks pesados (Node.js/React/Vue), garantindo carregamento instantâneo, zero dependências externas extras e execução leve acoplada ao próprio FastAPI (`StaticFiles`). |
| **Hospedagem / Deploy** | **Render** | Permite deploy contínuo direto do GitHub, integração nativa de variáveis de ambiente seguras e plano gratuito estável. |

---

## 3. Contrato da API

### `GET /health`
Verifica a disponibilidade da aplicação (*liveness probe*). **Não exige chave de API.**
* **Código HTTP:** `200 OK`
* **Exemplo de Resposta:**
  ```json
  {
    "status": "ok",
    "versao": "v1",
    "ambiente": "dev"
  }
  ```

### `POST /predict`
Realiza a inferência da mensagem.
* **Autenticação:** Se a variável `API_KEY` estiver configurada no servidor, exige o cabeçalho `X-API-Key`.
* **Corpo da Requisição (Payload):**
  ```json
  {
    "texto": "Parabéns! Você ganhou um prêmio, clique no link para resgatar agora"
  }
  ```
* **Códigos de Resposta:**
  * `200 OK`: Classificação realizada com sucesso.
    ```json
    {
      "rotulo": "spam",
      "probabilidade": 0.67,
      "versao": "v1"
    }
    ```
  * `400 Bad Request`: Mensagem vazia ou apenas com espaços em branco.
  * `401 Unauthorized`: Cabeçalho `X-API-Key` ausente ou incorreto.

### `GET /metrics`
Endpoint de telemetria para o painel de operação em tempo real.
* **Exemplo de Resposta:**
  ```json
  {
    "requisicoes": 128,
    "erros_5xx": 0,
    "latencia_media_ms": 14.2
  }
  ```

---

## 4. Variáveis de Ambiente e Configurações

Seguindo os princípios do **12-Factor App**, nenhuma configuração sensível ou de ambiente é gravada de forma fixa no código-fonte.

As variáveis devem ser configuradas via arquivo `.env` (no ambiente local) ou no painel da plataforma de deploy:

| Variável | Descrição | Exemplo Local (`dev`) | Exemplo Produção (`prod`) |
|---|---|---|---|
| `APP_ENV` | Define o ambiente em que a aplicação está executando | `dev` | `prod` ou `test` |
| `MODEL_VERSION` | Versão ativa do modelo de ML para rastreabilidade | `v1` | `v1` |
| `API_KEY` | Chave secreta exigida para autorizar o `/predict` | `chave-ficticia-123` | *(definida no Render)* |
| `MODEL_PATH` | *(Opcional)* Caminho para o arquivo `.joblib` | `model/modelo.joblib` | `model/modelo.joblib` |

> ⚠️ **Segurança:** O arquivo `.env` está explicitamente ignorado no `.gitignore` para garantir que nenhum segredo seja enviado ao repositório público.

---

## 5. Como Executar Localmente Passo a Passo

### 5.1 Pré-requisitos
* **Python:** Versão 3.10, 3.11 ou 3.12 instalada.
* **Git** instalado.

### 5.2 Passo a Passo

1. **Clonar o Repositório:**
   ```bash
   git clone https://github.com/DK7E3/mlops-spam-Rafaela-Francisco.git
   cd mlops-spam-Rafaela-Francisco
   ```

2. **Criar e Ativar o Ambiente Virtual:**
   * No Linux / macOS:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   * No Windows (PowerShell):
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```

3. **Instalar as Dependências:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configurar as Variáveis de Ambiente:**
   Copie o modelo de exemplo para criar seu arquivo `.env`:
   * No Linux / macOS: `cp .env.example .env`
   * No Windows: `copy .env.example .env`

5. **Iniciar a Aplicação:**
   ```bash
   uvicorn app.main:app --reload
   ```

6. **Acessar a Aplicação:**
   * **Interface Web:** [http://localhost:8000](http://localhost:8000)
   * **Documentação Interativa Swagger:** [http://localhost:8000/docs](http://localhost:8000/docs)
   * **Healthcheck:** [http://localhost:8000/health](http://localhost:8000/health)

---

## 6. Exemplos de Teste via Terminal (`curl`)

### Teste de Healthcheck:
```bash
curl -X GET http://localhost:8000/health
```

### Teste de Classificação com Chave Correta:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: chave-ficticia-123" \
  -d "{\"texto\": \"Você ganhou um prêmio exclusivo! Clique aqui para resgatar\"}"
```

### Teste de Rejeição de Chave Inválida (HTTP 401):
```bash
curl -i -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: chave-incorreta" \
  -d "{\"texto\": \"Olá, tudo bem?\"}"
```

### Teste de Rejeição de Mensagem Vazia (HTTP 400):
```bash
curl -i -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: chave-ficticia-123" \
  -d "{\"texto\": \"   \"}"
```

---

## 7. Divisão de Responsabilidades da Dupla

O desenvolvimento foi conduzido em regime de **Pair Programming** (piloto e copiloto), com alternância de tarefas:

* **Francisco:**
  * Configuração inicial do ambiente virtual e dependências exatas (`scikit-learn==1.8.0`).
  * Implementação da arquitetura base do FastAPI e carregamento do modelo em memória.
  * Construção e estilização da interface web (`index.html`) com tema Dark Mode.
  * Criação do coletor de métricas e gerador de carga (50 requisições).

* **Rafaela:**
  * Implementação da lógica de autenticação por cabeçalho `X-API-Key` e validações de erro (HTTP 401 e 400).
  * Configuração da política de CORS para garantir integridade das chamadas da interface.
  * Elaboração dos testes unitários e de integração dos endpoints.
  * Estruturação e redação da documentação técnica e instruções do `README.md`.
