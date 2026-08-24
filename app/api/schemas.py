from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class BenchmarkRunRequest(BaseModel):
    dataset_name: str
    model_configs: List[Dict[str, Any]]

class BenchmarkResponse(BaseModel):
    run_id: str
    status: str
    message: str

class ReportResponse(BaseModel):
    run_id: str
    url: str
