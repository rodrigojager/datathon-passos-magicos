# Datathon Passos Mágicos

Solução completa de Machine Learning Engineering para estimar o risco de defasagem escolar futura de estudantes da Associação Passos Mágicos.

## Resumo

O projeto foi estruturado como uma aplicação FastAPI com dashboard embutido, pipeline de treinamento em `scikit-learn`, persistência operacional em PostgreSQL e empacotamento com Docker.

Fluxo principal:

1. arquivos CSV entram pela API
2. o schema de entrada é detectado automaticamente
3. os dados são normalizados para um schema canônico
4. a base consolidada é persistida no PostgreSQL com deduplicação por `ra + ano_referencia`
5. o treino consome a tabela consolidada
6. predições, métricas, drift e eventos ficam registrados no banco
7. a predição devolve a decisão binária principal e scores analógicos complementares
8. relatórios exportáveis são gerados a partir dessas execuções persistidas

## Arquitetura

- API e dashboard: FastAPI
- ML: pandas, numpy, scikit-learn
- Banco operacional: PostgreSQL
- ORM/acesso a dados: SQLAlchemy
- Serialização do modelo: joblib
- Testes: pytest + pytest-cov
- Infraestrutura: Docker + docker compose

## Modelos Comparados

- `baseline_persistencia`
- `logistic_regression`
- `random_forest`
- `hist_gradient_boosting`

O modelo estatístico padrão da API atualmente selecionado é:

- `random_forest`
- `strategy_b`

O baseline permanece no monitoramento e na análise comparativa porque é uma régua importante de negócio.

Os demais candidatos treinados também ficam disponíveis para uso manual na tela de `Predição`, por meio do dropdown de seleção de modelo.

## Estrutura do Repositório

```text
datathon-entrega/
├── app/
│   ├── main.py
│   ├── static/
│   └── templates/
├── dados/
├── docs/
│   ├── chapters/
│   └── README.md
├── instrucoes/
├── src/
│   ├── config.py
│   ├── baseline_analysis.py
│   ├── dashboard_data.py
│   ├── database.py
│   ├── db_models.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   ├── monitoring.py
│   ├── normalization.py
│   ├── schema_registry.py
│   ├── storage.py
│   ├── transition_scoring.py
│   └── utils.py
├── tests/
├── Dockerfile
├── docker-compose.yml
├── docker-compose.vps.yml
├── .env.example
├── .env.vps.example
├── pytest.ini
├── requirements.txt
└── README.md
```

Diretórios gerados em execução, e por isso não versionados:

- `app/model/`
- `artifacts/`
- `data/`

## Execução com Docker Compose

1. Copie o arquivo de exemplo:

```bash
cp .env.example .env
```

2. Ajuste as variáveis do `.env`.

3. Suba aplicação e banco:

```bash
docker compose up --build
```

Serviços:

- app: `http://127.0.0.1:8000`
- postgres: `localhost:5432`

Variáveis esperadas no `.env`:

```text
COMPOSE_APP_CONTAINER
COMPOSE_POSTGRES_CONTAINER
APP_PORT
POSTGRES_PORT
POSTGRES_DB
POSTGRES_USER
POSTGRES_PASSWORD
DATABASE_URL
ROOT_PATH
```

Exemplo de `DATABASE_URL` local:

```text
postgresql+psycopg://datathon:SUA_SENHA@postgres:5432/datathon
```

Passo a passo local mínimo:

```powershell
Copy-Item .env.example .env
docker compose up -d --build
curl http://127.0.0.1:8000/health
curl -X POST "http://127.0.0.1:8000/data/bootstrap"
curl -X POST "http://127.0.0.1:8000/train"
Start-Process "http://127.0.0.1:8000/dashboard"
```

## Execução Local

Instalar dependências:

```bash
pip install -r requirements.txt
```

Definir `DATABASE_URL` apontando para o PostgreSQL da sua máquina ou usar diretamente o `docker compose` do projeto.

Subir a API:

```bash
uvicorn app.main:app --reload
```

Se optar por rodar sem Docker, o PostgreSQL precisa estar acessível e a variável `DATABASE_URL` precisa apontar para ele corretamente.

## Endpoints Principais

- `GET /health`
- `POST /data/normalize`
- `POST /data/ingest`
- `POST /data/bootstrap`
- `GET /data/status`
- `POST /train`
- `POST /predict`
- `POST /monitor/drift`
- `GET /monitor/actionable-insights`
- `GET /monitor/baseline-analysis`
- `GET /monitor/transition-scores`
- `GET /monitor/metrics`
- `GET /monitor/logs`
- `GET /monitor/models/comparison`
- `GET /monitor/drift/summary`
- `GET /monitor/drift/report`

Dashboard:

- `/dashboard`
- `/dashboard/models`
- `/dashboard/predict`
- `/dashboard/drift`
- `/dashboard/logs`
- `/dashboard/docs`

## Publicação

Se a aplicação for publicada atrás de um prefixo de URL:

```text
ROOT_PATH=/seu_basepath
```

## Exemplos de Uso

Bootstrap dos CSVs brutos já existentes em `dados/`:

```bash
curl -X POST "http://127.0.0.1:8000/data/bootstrap"
```

Treino:

```bash
curl -X POST "http://127.0.0.1:8000/train"
```

Predição:

```bash
curl -X POST "http://127.0.0.1:8000/predict?reference_year=2024&selected_model_id=random_forest__strategy_b" ^
  -F "file=@dados/BASE DE DADOS - DATATHON - PEDE2024.csv"
```

O `/predict` devolve:

- classe binária principal
- probabilidade geral de risco
- grupo atual segundo o baseline
- score de entrada em risco para quem hoje não está defasado
- score de recuperação para quem hoje já está defasado
- score de persistência da defasagem
- identificador do modelo principal selecionado pelo usuário

Drift:

```bash
curl -X POST "http://127.0.0.1:8000/monitor/drift?reference_year=2024" ^
  -F "file=@dados/BASE DE DADOS - DATATHON - PEDE2024.csv"
```

Análise residual do baseline:

```bash
curl -X GET "http://127.0.0.1:8000/monitor/baseline-analysis"
```

Scores de transição:

```bash
curl -X GET "http://127.0.0.1:8000/monitor/transition-scores"
```

## Testes

```bash
pytest -q
```

Cobertura mínima exigida:

- `80%`

## Documentação

Documentação disponível no repositório:

- [docs/README.md](docs/README.md)
- [docs/chapters](docs/chapters)
