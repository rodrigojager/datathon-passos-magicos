## 6. Operação da Solução

### 6.1. Como subir localmente do zero

Este é o passo a passo recomendado para rodar a solução em ambiente local, usando Windows com PowerShell e Docker Desktop.

#### Pré-requisitos

Antes de começar, o ambiente precisa ter:

- `Python 3.12` ou compatível instalado
- `Docker Desktop` instalado e em execução
- `Git`, `PowerShell` ou terminal equivalente

#### Passo 1. Abrir a pasta do projeto

No terminal:

```powershell
cd caminho\para\datathon-entrega
```

Ou, de forma mais genérica:

- clone o repositório
- abra um terminal na raiz do projeto
- execute os comandos a partir dessa pasta

#### Passo 2. Criar o arquivo `.env`

Copie o arquivo de exemplo:

```powershell
Copy-Item .env.example .env
```

O arquivo `.env` precisa conter, no mínimo:

```text
COMPOSE_APP_CONTAINER=datathon-app
COMPOSE_POSTGRES_CONTAINER=datathon-postgres
APP_PORT=8000
POSTGRES_PORT=5432
POSTGRES_DB=datathon
POSTGRES_USER=datathon
POSTGRES_PASSWORD=SUA_SENHA_AQUI
DATABASE_URL=postgresql+psycopg://datathon:SUA_SENHA_AQUI@postgres:5432/datathon
ROOT_PATH=
```

Observações:

- `POSTGRES_PASSWORD` e a senha dentro de `DATABASE_URL` precisam ser iguais
- em execução local, `ROOT_PATH` deve ficar vazio
- o hostname `postgres` na `DATABASE_URL` funciona porque a aplicação e o banco sobem na mesma rede do `docker compose`

#### Passo 3. Subir PostgreSQL e aplicação com Docker Compose

Execute:

```powershell
docker compose up -d --build
```

Esse comando sobe:

- o container `postgres`
- o container `app`

Se quiser acompanhar os logs:

```powershell
docker compose logs -f
```

Se quiser parar tudo:

```powershell
docker compose down
```

#### Passo 4. Verificar se a API está online

Abra no navegador:

- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/dashboard`
- `http://127.0.0.1:8000/docs`

Com `curl`:

```powershell
curl http://127.0.0.1:8000/health
```

O esperado é:

```json
{"status":"ok"}
```

#### Passo 5. Popular a base inicial

Com a API no ar, carregue os CSVs históricos do projeto:

```powershell
curl -X POST "http://127.0.0.1:8000/data/bootstrap"
```

Esse passo:

- lê os arquivos da pasta `dados/`
- detecta o schema de cada um
- normaliza para o schema canônico
- deduplica por `ra + ano_referencia`
- grava o consolidado no PostgreSQL

#### Passo 6. Conferir o estado da base

```powershell
curl "http://127.0.0.1:8000/data/status"
```

O esperado é ver:

- quantidade de linhas carregadas
- quantidade de alunos únicos
- anos disponíveis na base

#### Passo 7. Treinar os modelos

```powershell
curl -X POST "http://127.0.0.1:8000/train"
```

Esse passo:

- consulta a base consolidada no PostgreSQL
- monta os pares temporais de treino
- compara baseline e modelos estatísticos
- seleciona o modelo padrão da API
- gera bundles e relatórios

#### Passo 8. Validar a predição

Exemplo com o arquivo de 2024:

```powershell
curl -X POST "http://127.0.0.1:8000/predict?reference_year=2024&selected_model_id=random_forest__strategy_b" `
  -F "file=@dados/BASE DE DADOS - DATATHON - PEDE2024.csv"
```

O retorno inclui:

- decisão binária principal
- probabilidade geral de risco
- grupo do estudante
- score de entrada em risco ou recuperação

#### Passo 9. Validar monitoramento

Drift:

```powershell
curl -X POST "http://127.0.0.1:8000/monitor/drift?reference_year=2024" `
  -F "file=@dados/BASE DE DADOS - DATATHON - PEDE2024.csv"
```

Modelos:

```powershell
curl "http://127.0.0.1:8000/monitor/models/comparison"
```

Logs:

```powershell
curl "http://127.0.0.1:8000/monitor/logs?limit=20"
```

#### Passo 10. Rodar os testes

Se quiser validar o projeto localmente antes de usar:

```powershell
pytest -q
```

#### Fluxo mínimo recomendado em ambiente local

Em ordem:

1. criar `.env`
2. executar `docker compose up -d --build`
3. chamar `/health`
4. chamar `/data/bootstrap`
5. chamar `/train`
6. abrir `/dashboard`
7. testar `/predict`

### 6.2. Fluxo operacional do zero

Para usar a solução a partir de um ambiente vazio, a sequência recomendada é esta:

1. iniciar a API
2. popular a base consolidada no PostgreSQL
3. treinar os modelos
4. usar predição, comparação entre modelos e drift
5. acompanhar o dashboard e os logs

Em termos práticos, a ordem correta é:

- `POST /data/bootstrap` ou `POST /data/ingest`
- `POST /train`
- `POST /predict`
- `POST /monitor/drift`

### 6.3. Como popular o banco de dados pela API

Há duas formas principais.

#### Carga inicial com os arquivos do projeto

Esse é o caminho mais simples para montar a base histórica inicial:

```bash
curl -X POST "http://127.0.0.1:8000/data/bootstrap"
```

O que esse endpoint faz:

- lê os CSVs da pasta `dados/`
- detecta o schema de cada arquivo
- normaliza todos para o schema canônico
- deduplica usando `ra + ano_referencia`
- grava os registros válidos no PostgreSQL
- registra a execução de ingestão

#### Carga incremental com um arquivo enviado manualmente

Esse é o caminho para anos novos, como `2025`, ou para reaproveitar a API com novos arquivos compatíveis:

```bash
curl -X POST "http://127.0.0.1:8000/data/ingest?reference_year=2025" ^
  -F "file=@meu_arquivo_2025.csv"
```

O que esse endpoint faz:

- recebe um arquivo CSV
- detecta o schema
- normaliza colunas e tipos
- valida o ano de referência
- verifica se cada linha já existe
- insere somente registros novos

Se você quiser apenas baixar o arquivo já padronizado, sem inserir no banco:

```bash
curl -X POST "http://127.0.0.1:8000/data/normalize" ^
  -F "file=@meu_arquivo_2025.csv" ^
  --output arquivo_normalizado.csv
```

### 6.4. Como gerar os modelos pela API

Depois que o banco já tiver sido populado com a base consolidada, o próximo passo é treinar:

```bash
curl -X POST "http://127.0.0.1:8000/train"
```

Esse endpoint:

- consulta a base consolidada no PostgreSQL
- monta os pares temporais de treino e validação
- compara baseline e modelos estatísticos
- seleciona o melhor bundle estatístico
- treina também a camada complementar de transição
- grava os resultados no banco
- gera os bundles e exportações de treino

Importante:

- o treino **não** acontece automaticamente quando a API inicia
- ele só acontece quando `/train` é chamado

### 6.5. O que a API entrega depois do treino

Depois do `/train`, a solução fica pronta para:

- prever risco futuro com `/predict`
- consultar a comparação de modelos no painel `Modelos`
- gerar relatórios de drift com `/monitor/drift`
- alimentar o dashboard com métricas, eventos e resumos

### 6.6. Quando os artefatos são gerados

Os artefatos são gerados sob demanda, conforme cada endpoint é executado.

Eles **não** aparecem automaticamente só porque a API subiu.

Exemplos:

- `/train` gera:
  - `app/model/student_risk_model.joblib`
  - `app/model/candidate_models.joblib`
  - `app/model/validation_candidate_models.joblib`
  - `artifacts/exports/reports/training_report.json`
  - `artifacts/exports/reports/baseline_analysis.json`
  - `artifacts/exports/reports/transition_scores.json`

- `/monitor/drift` gera:
  - `artifacts/exports/reports/drift_summary.json`
  - `artifacts/exports/reports/drift_report.html`

- uso geral da aplicação gera:
  - `artifacts/logs/app.log`

### 6.7. O que esperar ao iniciar uma base vazia

Se o banco acabou de ser limpo, o comportamento esperado é:

- o dashboard abrir, mas sem métricas históricas úteis
- `/data/status` mostrar base vazia
- `/train` falhar até que existam dados suficientes no banco
- `/predict` falhar até existir um bundle treinado
- `/monitor/drift` falhar até existir uma referência de treino

Esse comportamento é esperado e correto. A ordem natural é:

1. carregar dados
2. treinar
3. prever e monitorar
### 6.8. Endpoints implementados

#### Saúde

- `GET /health`

#### Dados

- `POST /data/normalize`
- `POST /data/ingest`
- `POST /data/bootstrap`
- `GET /data/status`

#### Machine Learning

- `POST /train`
- `POST /predict`

#### Monitoramento

- `POST /monitor/drift`
- `GET /monitor/actionable-insights`
- `GET /monitor/baseline-analysis`
- `GET /monitor/transition-scores`
- `GET /monitor/metrics`
- `GET /monitor/logs`
- `GET /monitor/models/comparison`
- `GET /monitor/drift/summary`
- `GET /monitor/drift/report`

### 6.9. Dashboard

O dashboard é servido pela mesma aplicação e possui as páginas:

- `/dashboard`
- `/dashboard/models`
- `/dashboard/predict`
- `/dashboard/drift`
- `/dashboard/logs`

O dashboard consulta o banco para exibir:

- estado da base consolidada
- último treino
- comparação entre modelos
- resumos operacionais e comparações recentes
- drift mais recente
- eventos operacionais

### 6.10. Exemplos de uso da API

Base local de exemplo:
- `http://127.0.0.1:8000`

Verificação de saúde:

```bash
curl -X GET "http://127.0.0.1:8000/health"
```

Normalização:

```bash
curl -X POST "http://127.0.0.1:8000/data/normalize" ^
  -F "file=@dados/BASE DE DADOS - DATATHON - PEDE2024.csv" ^
  --output arquivo_normalizado.csv
```

Bootstrap da base:

```bash
curl -X POST "http://127.0.0.1:8000/data/bootstrap"
```

Ingestão incremental:

```bash
curl -X POST "http://127.0.0.1:8000/data/ingest?reference_year=2025" ^
  -F "file=@meu_arquivo_2025.csv"
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

Além da classe principal, essa chamada pode devolver:

- probabilidade geral de risco
- grupo atual segundo o baseline
- score de entrada em risco
- score de recuperação
- score de persistência da defasagem
- identificador do modelo principal escolhido na chamada

Drift:

```bash
curl -X POST "http://127.0.0.1:8000/monitor/drift?reference_year=2024" ^
  -F "file=@dados/BASE DE DADOS - DATATHON - PEDE2024.csv"
```

Métricas e logs:

```bash
curl -X GET "http://127.0.0.1:8000/monitor/metrics"
```

```bash
curl -X GET "http://127.0.0.1:8000/monitor/logs?limit=50"
```

```bash
curl -X GET "http://127.0.0.1:8000/monitor/baseline-analysis"
```

```bash
curl -X GET "http://127.0.0.1:8000/monitor/actionable-insights"
```

```bash
curl -X GET "http://127.0.0.1:8000/monitor/transition-scores"
```

### 6.11. Relatórios exportáveis

As exportações são geradas a partir das execuções persistidas no banco.

Principais saídas:

- `artifacts/exports/reports/training_report.json`
- `artifacts/exports/reports/transition_scores.json`
- `artifacts/exports/reports/drift_summary.json`
- `artifacts/exports/reports/drift_report.html`

### 6.12. Docker, Compose e deploy

O projeto inclui:

- `Dockerfile` para a aplicação
- `docker-compose.yml` com dois serviços: `app` e `postgres`

Subida local:

```bash
docker compose up --build
```

Conexão usada pela aplicação no compose:

- `postgresql+psycopg://datathon:datathon@postgres:5432/datathon`

Quando a aplicação é publicada atrás de um prefixo de URL, a variável `ROOT_PATH` pode ser usada para manter links, assets e chamadas internas coerentes com o basepath publicado.

### 6.13. Testes

O projeto possui testes automatizados cobrindo:

- detecção de schema
- normalização
- deduplicação e ingestão
- treino
- predição
- avaliação
- drift
- dashboard
- API

Execução:

```bash
pytest -q
```

Cobertura mínima exigida:

- `80%`

Cobertura atual:

- superior a `89%`

---
