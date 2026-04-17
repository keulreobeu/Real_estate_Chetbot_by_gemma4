from __future__ import annotations

from pydantic import BaseModel, Field, constr


class AskRequest(BaseModel):
    question: constr(strip_whitespace=True, min_length=1)


class RegionGroup(BaseModel):
    label: str
    districts: list[str]


class GenerationReadyResponse(BaseModel):
    ready: bool
    status: str
    backend: str
    model_id: str
    device_map: str = ""
    model_source: str = ""
    load_runtime_ms: int = 0
    generate_ms: int = 0
    probe_total_ms: int = 0
    text_preview: str = ""
    error: str = ""
    checked_at: int = 0


class AskResponse(BaseModel):
    question: str
    answer: str
    answer_type: str
    match_status: str
    query_type: str
    cited_doc_ids: list[str]
    top_doc_id: str
    retrieval_score: float
    used_fields: list[str]
    data_cutoff: str
    limitations: list[str]
    backend: str
    model_id: str
    runtime: str
    latency_ms: int
    finish_reason: str
    device_map: str = ""
    model_source: str = ""
    model_device: str = ""
    last_load_ms: int = 0
    last_generate_ms: int = 0
    local_files_only: bool = False


class StatusResponse(BaseModel):
    backend: str
    model_id: str
    runtime: str
    device_map: str = ""
    model_source: str = ""
    last_load_ms: int = 0
    request_timeout_seconds: int
    probe_error: str = ""
    rule_check_question: str
    rule_check_description: str
    generation_check_description: str
    generation_ready: bool
    generation_probe_supported: bool
    generation_probe_in_progress: bool
    last_generation_probe: GenerationReadyResponse
    regions: list[RegionGroup]
    server_started_at: str
    pid: int
    port: int


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Structured server-side error message.")
