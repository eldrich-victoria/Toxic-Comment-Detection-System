from fastapi import APIRouter, BackgroundTasks, HTTPException
from app.api.schemas import BenchmarkRunRequest, BenchmarkResponse
from app.core.logger import logger
import time

router = APIRouter()

@router.post("/benchmark/run", response_model=BenchmarkResponse)
async def run_benchmark(request: BenchmarkRunRequest, background_tasks: BackgroundTasks):
    run_id = f"run_{int(time.time())}"
    logger.info(f"Starting benchmark run: {run_id}")
    # In production, dispatch background task to BenchmarkRunner
    return BenchmarkResponse(run_id=run_id, status="STARTED", message="Benchmark dispatched successfully.")

@router.get("/benchmark/results")
async def get_results():
    return {"status": "success", "data": []}

@router.get("/benchmark/report/{run_id}")
async def get_report(run_id: str):
    return {"run_id": run_id, "status": "success"}

@router.get("/benchmark/charts/{run_id}")
async def get_charts(run_id: str):
    return {"run_id": run_id, "status": "success"}

@router.get("/benchmark/models")
async def get_models():
    return {"models": ["baseline_logistic", "distilbert_toxic"]}

@router.get("/benchmark/rankings")
async def get_rankings():
    return {"rankings": {}}

@router.get("/benchmark/history")
async def get_history():
    return {"history": []}
