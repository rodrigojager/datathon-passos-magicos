from __future__ import annotations

from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.config import ROOT_DIR, ROOT_PATH_PREFIX, ensure_directories
from src.baseline_analysis import build_actionable_insights, read_latest_baseline_analysis
from src.database import init_db
from src.dashboard_data import (
    build_overview_metrics,
    read_latest_drift_report_html,
    read_drift_summary,
    read_recent_logs,
    read_training_report,
)
from src.monitoring import MonitoringError, generate_drift_report
from src.modeling import (
    ModelingError,
    candidate_model_id,
    predict_from_normalized_dataframe,
    train_and_select_model,
)
from src.normalization import (
    NormalizationError,
    dataframe_to_csv_bytes,
    normalize_uploaded_csv,
    normalized_output_name,
)
from src.storage import (
    bootstrap_raw_directory,
    get_data_status_summary,
    ingest_normalized_dataframe,
)
from src.transition_scoring import read_latest_transition_score_summary
from src.utils import get_logger


OPENAPI_TAGS = [
    {
        "name": "Infraestrutura",
        "description": "Rotas de disponibilidade e verificação básica da aplicação.",
    },
    {
        "name": "Dados",
        "description": "Rotas para normalizar arquivos, ingerir novos lotes e consultar o estado da base consolidada.",
    },
    {
        "name": "Modelagem",
        "description": "Rotas para treinar o modelo principal, consultar candidatos comparados e gerar previsoes para novos lotes.",
    },
    {
        "name": "Monitoramento",
        "description": "Rotas para acompanhar métricas operacionais, logs, drift e análises comparativas dos modelos.",
    },
]


app = FastAPI(
    title="Datathon Passos Mágicos API",
    version="0.1.0",
    description="API para normalização, ingestão, treinamento, previsão e dashboard de risco de defasagem escolar.",
    root_path=ROOT_PATH_PREFIX,
    docs_url=None,
    redoc_url=None,
    openapi_tags=OPENAPI_TAGS,
)
ensure_directories()
init_db()
logger = get_logger(__name__)

templates = Jinja2Templates(directory=str(ROOT_DIR / "app" / "templates"))
app.mount("/static", StaticFiles(directory=str(ROOT_DIR / "app" / "static")), name="static")

DOCS_ROOT = ROOT_DIR / "docs"
DOCS_CHAPTERS_DIR = DOCS_ROOT / "chapters"


def _read_markdown_title(markdown_path: Path) -> str:
    for line in markdown_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return markdown_path.stem.replace("_", " ").title()


def _build_docs_navigation() -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for path in sorted(DOCS_CHAPTERS_DIR.glob("*.md")):
        slug = path.stem
        title = _read_markdown_title(path)
        if slug == "01_visao_geral":
            title = "Introdução"
        items.append(
            {
                "slug": slug,
                "title": title,
                "href": f"/dashboard/docs/{slug}",
            }
        )
    reference_path = DOCS_ROOT / "01_referencia_de_arquivos_e_metodos.md"
    if reference_path.exists():
        items.append(
            {
                "slug": "referencia_de_arquivos_e_metodos",
                "title": "Referência de Arquivos e Métodos",
                "href": "/dashboard/docs/referencia_de_arquivos_e_metodos",
            }
        )
    return items


DOCS_MENU_ITEMS = _build_docs_navigation()


def _resolve_markdown_doc(doc_slug: str) -> tuple[Path, str]:
    if doc_slug == "referencia_de_arquivos_e_metodos":
        path = DOCS_ROOT / "01_referencia_de_arquivos_e_metodos.md"
    else:
        path = DOCS_CHAPTERS_DIR / f"{doc_slug}.md"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Documento não encontrado.")
    title = _read_markdown_title(path)
    if doc_slug == "01_visao_geral":
        title = "Introdução"
    return path, title


def _normalize_or_422(upload: UploadFile, reference_year: int | None):
    try:
        content = upload.file.read()
        return normalize_uploaded_csv(
            content=content,
            filename=upload.filename or "arquivo.csv",
            reference_year=reference_year,
        )
    except NormalizationError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


def _render_dashboard(
    request: Request,
    template_name: str,
    page_title: str,
    extra_context: dict[str, object] | None = None,
):
    base_path = request.scope.get("root_path", "") or ""
    context = {
        "request": request,
        "page_title": page_title,
        "app_title": app.title,
        "base_path": base_path,
        "docs_menu_items": DOCS_MENU_ITEMS,
        "active_doc_slug": None,
    }
    if extra_context:
        context.update(extra_context)
    return templates.TemplateResponse(
        request,
        template_name,
        context,
    )


@app.get(
    "/health",
    tags=["Infraestrutura"],
    summary="Verificar disponibilidade",
    description="Confirma se a aplicação está online e pronta para receber requisições.",
)
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root_dashboard(request: Request):
    return _render_dashboard(request, "dashboard_overview.html", "Visão Geral")


@app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
def dashboard_overview(request: Request):
    return _render_dashboard(request, "dashboard_overview.html", "Visão Geral")


@app.get("/dashboard/models", response_class=HTMLResponse, include_in_schema=False)
def dashboard_models(request: Request):
    return _render_dashboard(request, "dashboard_models.html", "Comparação de Modelos")


@app.get("/dashboard/predict", response_class=HTMLResponse, include_in_schema=False)
def dashboard_predict(request: Request):
    return _render_dashboard(request, "dashboard_predict.html", "Predição por Arquivo")

@app.get("/dashboard/drift", response_class=HTMLResponse, include_in_schema=False)
def dashboard_drift(request: Request):
    return _render_dashboard(request, "dashboard_drift.html", "Drift")


@app.get("/dashboard/logs", response_class=HTMLResponse, include_in_schema=False)
def dashboard_logs(request: Request):
    return _render_dashboard(request, "dashboard_logs.html", "Logs")


@app.get("/dashboard/docs", response_class=HTMLResponse, include_in_schema=False)
def dashboard_docs_index(request: Request):
    if not DOCS_MENU_ITEMS:
        raise HTTPException(status_code=404, detail="Nenhum documento Markdown foi encontrado.")
    base_path = request.scope.get("root_path", "") or ""
    return RedirectResponse(url=f"{base_path}{DOCS_MENU_ITEMS[0]['href']}")


@app.get("/dashboard/docs/{doc_slug}", response_class=HTMLResponse, include_in_schema=False)
def dashboard_docs_markdown(request: Request, doc_slug: str):
    markdown_path, page_title = _resolve_markdown_doc(doc_slug)
    return _render_dashboard(
        request,
        "dashboard_markdown.html",
        page_title,
        extra_context={
            "active_doc_slug": doc_slug,
            "markdown_content": markdown_path.read_text(encoding="utf-8"),
            "markdown_source_path": str(markdown_path.relative_to(ROOT_DIR)),
        },
    )


@app.get("/docs", response_class=HTMLResponse, include_in_schema=False)
def docs_dashboard(request: Request):
    return _render_dashboard(request, "dashboard_docs.html", "Swagger")


@app.get("/swagger-ui", include_in_schema=False)
def swagger_ui(request: Request):
    base_path = request.scope.get("root_path", "") or ""
    return get_swagger_ui_html(
        openapi_url=f"{base_path}{app.openapi_url}",
        title=f"{app.title} - Swagger UI",
    )


@app.get(
    "/monitor/metrics",
    tags=["Monitoramento"],
    summary="Consultar métricas consolidadas",
    description="Retorna o resumo operacional usado no dashboard, incluindo dados carregados, treino recente e status geral.",
)
def monitor_metrics() -> dict[str, object]:
    return build_overview_metrics()


@app.get(
    "/monitor/logs",
    tags=["Monitoramento"],
    summary="Ler logs recentes",
    description="Retorna os eventos operacionais mais recentes registrados pela aplicação, com limite configurável.",
)
def monitor_logs(limit: int = Query(default=200, ge=1, le=1000)) -> dict[str, object]:
    logs = read_recent_logs(limit=limit)
    return {"count": len(logs), "logs": logs}


@app.get(
    "/monitor/models/comparison",
    tags=["Monitoramento"],
    summary="Comparar candidatos de treino",
    description="Expõe o relatório do último treinamento com os modelos testados, estratégias avaliadas e o candidato selecionado.",
)
def monitor_models_comparison() -> dict[str, object]:
    training = read_training_report()
    candidates = training.get("candidates", [])
    best_candidate_id = None
    if candidates:
        sorted_candidates = sorted(
            candidates,
            key=lambda item: item.get("metrics", {}).get("f2", 0) or 0,
            reverse=True,
        )
        best_candidate = sorted_candidates[0]
        best_candidate_id = candidate_model_id(best_candidate["model_name"], best_candidate["strategy"])

    enriched_candidates = []
    for candidate in candidates:
        model_id = candidate_model_id(candidate["model_name"], candidate["strategy"])
        enriched_candidate = {
            **candidate,
            "model_id": model_id,
            "is_best_overall": model_id == best_candidate_id,
        }
        enriched_candidates.append(enriched_candidate)

    return {
        "selected_model": training.get("selected_model"),
        "selected_strategy": training.get("selected_strategy"),
        "selected_model_id": (
            candidate_model_id(training["selected_model"], training["selected_strategy"])
            if training.get("selected_model") and training.get("selected_strategy")
            else None
        ),
        "candidates": enriched_candidates,
    }


@app.get(
    "/monitor/baseline-analysis",
    tags=["Monitoramento"],
    summary="Ler análise residual do baseline",
    description="Mostra onde o baseline acertou ou falhou, ajudando a entender entrada em risco e recuperação.",
)
def monitor_baseline_analysis() -> dict[str, object]:
    analysis = read_latest_baseline_analysis()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analise do baseline ainda nao foi gerada.")
    return analysis


@app.get(
    "/monitor/actionable-insights",
    tags=["Monitoramento"],
    summary="Ler sinais acionáveis de piora e melhora",
    description="Retorna apenas variáveis mais acionáveis para intervenção, separando sinais mais associados a piora e recuperação.",
)
def monitor_actionable_insights() -> dict[str, object]:
    analysis = read_latest_baseline_analysis()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analise acionavel ainda nao foi gerada.")
    return build_actionable_insights(analysis)


@app.get(
    "/monitor/transition-scores",
    tags=["Monitoramento"],
    summary="Consultar scores de transição",
    description="Retorna os resultados dos modelos segmentados que estimam entrada em risco, recuperação e persistência.",
)
def monitor_transition_scores() -> dict[str, object]:
    summary = read_latest_transition_score_summary()
    if not summary:
        raise HTTPException(status_code=404, detail="Modelos de transicao ainda nao foram gerados.")
    return summary


@app.get(
    "/monitor/drift/summary",
    tags=["Monitoramento"],
    summary="Ler resumo de drift",
    description="Retorna o resumo mais recente da análise de drift, com alertas, quantidade de variáveis e versão dos dados.",
)
def monitor_drift_summary() -> dict[str, object]:
    return read_drift_summary()


@app.get(
    "/monitor/drift/report",
    tags=["Monitoramento"],
    summary="Abrir relatório HTML de drift",
    description="Entrega o relatório HTML mais recente de drift para inspeção visual detalhada.",
)
def monitor_drift_report():
    html_report = read_latest_drift_report_html()
    if not html_report:
        raise HTTPException(status_code=404, detail="Relatorio de drift ainda nao foi gerado.")
    return HTMLResponse(content=html_report)


@app.post(
    "/data/normalize",
    tags=["Dados"],
    summary="Normalizar um arquivo CSV",
    description="Recebe um CSV em schema cru, detecta o layout, aplica o schema canônico e devolve o arquivo normalizado.",
)
def normalize_file(
    file: UploadFile = File(
        ...,
        description="Arquivo CSV bruto em um layout compatível com 2022, 2023 ou 2024. Exemplo: PEDE2024.csv",
    ),
    reference_year: int | None = Query(
        default=None,
        description="Ano de referência a ser forçado quando o nome do arquivo não permitir inferência automática. Exemplo: 2024.",
        examples=[2024],
    ),
):
    result = _normalize_or_422(file, reference_year)
    logger.info("Normalizacao solicitada | arquivo=%s | schema=%s", result.source_filename, result.detected_schema)
    csv_content = dataframe_to_csv_bytes(result.dataframe)
    output_name = normalized_output_name(result.source_filename, result.reference_year)
    headers = {
        "X-Detected-Schema": result.detected_schema,
        "X-Reference-Year": str(result.reference_year),
        "X-Unknown-Columns": ",".join(result.unknown_columns),
        "Content-Disposition": f'attachment; filename="{output_name}"',
    }
    return StreamingResponse(
        BytesIO(csv_content),
        media_type="text/csv",
        headers=headers,
    )


@app.post(
    "/data/ingest",
    tags=["Dados"],
    summary="Ingerir um arquivo na base consolidada",
    description="Normaliza o arquivo recebido, deduplica registros e grava apenas as linhas novas no PostgreSQL.",
)
def ingest_file(
    file: UploadFile = File(
        ...,
        description="Arquivo CSV bruto que será normalizado e consolidado no PostgreSQL. Exemplo: novo_lote_2024.csv",
    ),
    reference_year: int | None = Query(
        default=None,
        description="Ano de referência usado para normalização quando o arquivo não explicita o ano no nome. Exemplo: 2024.",
        examples=[2024],
    ),
):
    result = _normalize_or_422(file, reference_year)
    summary = ingest_normalized_dataframe(result)
    logger.info("Ingestao solicitada | arquivo=%s", result.source_filename)
    return JSONResponse(content=summary.__dict__)


@app.post(
    "/data/bootstrap",
    tags=["Dados"],
    summary="Carregar os arquivos padrão do projeto",
    description="Executa a carga inicial a partir da pasta de dados configurada, consolidando os arquivos históricos no banco.",
)
def bootstrap_data() -> dict[str, object]:
    summaries = bootstrap_raw_directory()
    logger.info("Carga inicial solicitada | arquivos=%s", len(summaries))
    return {
        "files_processed": len(summaries),
        "summaries": [summary.__dict__ for summary in summaries],
        "data_status": get_data_status_summary(),
    }


@app.get(
    "/data/status",
    tags=["Dados"],
    summary="Consultar estado da base",
    description="Retorna quantidade de linhas, alunos únicos, anos carregados e versão atual da base consolidada.",
)
def data_status() -> dict[str, object]:
    return get_data_status_summary()


@app.post(
    "/train",
    tags=["Modelagem"],
    summary="Treinar e selecionar o modelo",
    description="Treina os candidatos definidos no projeto, compara métricas, escolhe o bundle operacional e salva os artefatos.",
)
def train_model() -> dict[str, object]:
    try:
        result = train_and_select_model()
    except ModelingError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    logger.info("Treinamento solicitado | modelo_selecionado=%s", result.selected_model)

    return {
        "selected_model": result.selected_model,
        "selected_strategy": result.selected_strategy,
        "metrics": result.metrics,
        "training_rows": result.training_rows,
        "validation_rows": result.validation_rows,
        "candidates": result.candidates,
        "training_run_id": result.training_run_id,
        "baseline_analysis_run_id": result.baseline_analysis_run_id,
        "transition_score_run_id": result.transition_score_run_id,
        "transition_scores": result.transition_scores,
        "data_version": result.data_version,
    }


@app.post(
    "/predict",
    tags=["Modelagem"],
    summary="Prever risco para um novo arquivo",
    description="Normaliza o arquivo enviado, aplica o modelo ativo e retorna a classe binária principal, probabilidades e scores complementares.",
)
def predict(
    file: UploadFile = File(
        ...,
        description="Arquivo CSV bruto com os alunos que devem receber previsão de risco futuro. Exemplo: PEDE2024.csv",
    ),
    reference_year: int | None = Query(
        default=None,
        description="Ano de referência do lote enviado. Informe quando o ano não puder ser inferido automaticamente. Exemplo: 2024.",
        examples=[2024],
    ),
    selected_model_id: str | None = Query(
        default=None,
        description="Identificador opcional do candidato a ser usado na predição. Exemplo: random_forest__strategy_b",
        examples=["random_forest__strategy_b"],
    ),
) -> dict[str, object]:
    result = _normalize_or_422(file, reference_year)
    try:
        predictions = predict_from_normalized_dataframe(
            result.dataframe,
            source_filename=result.source_filename,
            detected_schema=result.detected_schema,
            reference_year=result.reference_year,
            selected_model_id=selected_model_id,
        )
    except ModelingError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    logger.info("Predicao solicitada | arquivo=%s | linhas=%s", result.source_filename, len(predictions))

    return {
        "detected_schema": result.detected_schema,
        "reference_year": result.reference_year,
        "rows": len(predictions),
        "selected_model_id": (
            candidate_model_id(
                str(predictions["modelo_utilizado"].iloc[0]),
                str(predictions["estrategia_utilizada"].iloc[0]),
            )
            if not predictions.empty
            else selected_model_id
        ),
        "predictions": predictions.to_dict(orient="records"),
    }


@app.post(
    "/monitor/drift",
    tags=["Monitoramento"],
    summary="Gerar análise de drift",
    description="Compara um novo lote de dados com a referência do treino para detectar mudanças relevantes nas distribuições.",
)
def monitor_drift(
    file: UploadFile = File(
        ...,
        description="Arquivo CSV novo a ser comparado com a referência do treino para detecção de drift.",
    ),
    reference_year: int | None = Query(
        default=None,
        description="Ano de referência do lote enviado, quando necessário. Exemplo: 2025.",
        examples=[2025],
    ),
) -> dict[str, object]:
    result = _normalize_or_422(file, reference_year)
    try:
        drift = generate_drift_report(
            result.dataframe,
            source_filename=result.source_filename,
            detected_schema=result.detected_schema,
            reference_year=result.reference_year,
        )
    except MonitoringError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error

    logger.info(
        "Analise de drift solicitada | arquivo=%s | alertas=%s",
        result.source_filename,
        drift.alert_count,
    )
    return {
        "detected_schema": result.detected_schema,
        "reference_year": result.reference_year,
        "report_path": drift.report_path,
        "feature_rows": drift.feature_rows,
        "alert_count": drift.alert_count,
        "details": drift.details,
        "drift_run_id": drift.drift_run_id,
        "data_version": drift.data_version,
    }
