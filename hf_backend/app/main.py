"""
LLM 推理后端 FastAPI 入口
支持带 KV Cache 的迭代生成与预填充缓存
"""
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from uuid import uuid4
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .model_manager import ModelManager
from .inference_service import InferenceService
from .config import get_config
from .telemetry import JsonlTelemetryStore


# ---------- 配置 ----------
config = get_config()
MODEL_ID = config.get_model_id()
TORCH_DTYPE = config.get_torch_dtype()


# 全局服务（启动时加载）
model_manager: Optional[ModelManager] = None
inference_service: Optional[InferenceService] = None
telemetry_store: Optional[JsonlTelemetryStore] = None


# ---------- Pydantic 请求体 ----------
class PinYinConstraintGenerateRequest(BaseModel):
    prompt: str = Field(..., description="用户输入的文本prompt")
    pinyin_constraints: List[str] = Field(default_factory=list, description="与输入文本对应的拼音约束列表")
    client_request_id: Optional[str] = Field(default=None, description="可选的客户端请求 ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="可选的请求附加元数据")


class GenerateCompletionResponse(BaseModel):
    request_id: str
    responses: str
    candidates: List[str]
    latency_ms: float


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage] = Field(default_factory=list)
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


class ChatCompletionMessage(BaseModel):
    role: str = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionMessage
    finish_reason: str = "stop"


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage = Field(default_factory=ChatCompletionUsage)


class FeedbackRequest(BaseModel):
    request_id: Optional[str] = Field(default=None, description="对应预测请求的 request_id")
    prompt: str = Field(default="", description="原始上下文 prompt")
    pinyin_constraints: List[str] = Field(default_factory=list, description="预测时的拼音约束")
    shown_candidates: List[str] = Field(default_factory=list, description="展示给用户的候选列表")
    selected_candidate: Optional[str] = Field(default=None, description="用户最终接受的候选")
    accepted: bool = Field(default=True, description="用户是否接受了模型候选")
    expected_text: Optional[str] = Field(default=None, description="若用户未接受，可填写期望输出")
    rejected_candidates: List[str] = Field(default_factory=list, description="明确拒绝的候选")
    source: str = Field(default="user", description="反馈来源，如 user / telemetry / replay")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="附加元数据")


# ---------- 生命周期 ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_manager, inference_service, telemetry_store
    model_manager = ModelManager(
        model_id=MODEL_ID,
        torch_dtype=TORCH_DTYPE,
        device_map=config.get_device_map(),
        trust_remote_code=config.get_trust_remote_code(),
        adapter_path=config.get_adapter_path(),
        merge_adapter_on_load=config.should_merge_adapter_on_load(),
        model_kwargs=config.get_model_kwargs(),
    )
    model_manager.load_model()
    inference_service = InferenceService(
        model_manager=model_manager,
        use_chat_template=config.use_chat_template(),
    )
    telemetry_store = JsonlTelemetryStore(
        enabled=config.is_telemetry_enabled(),
        root_dir=config.get_telemetry_directory(),
    )
    yield
    model_manager = None
    inference_service = None
    telemetry_store = None


app = FastAPI(
    title="LLM Inference Backend",
    description="基于 transformers 的 LLM 推理后端，支持拼音约束、LoRA 适配器与反馈遥测",
    version="0.2.0",
    lifespan=lifespan,
)


def get_inference():
    if inference_service is None:
        raise HTTPException(status_code=503, detail="服务未就绪")
    return inference_service


def get_telemetry() -> JsonlTelemetryStore:
    if telemetry_store is None:
        raise HTTPException(status_code=503, detail="遥测服务未就绪")
    return telemetry_store


# ---------- 接口 ----------
@app.get("/health")
def health():
    model_info = model_manager.get_model_description() if model_manager else None
    return {
        "status": "ok",
        "model": model_info,
        "telemetry_enabled": bool(telemetry_store and telemetry_store.is_enabled()),
    }



@app.post("/v1/generate/completions")
def generate_completions(req: PinYinConstraintGenerateRequest) -> GenerateCompletionResponse:
    """使用Base模型进行生成补全"""
    start_time = time.time()
    request_id = req.client_request_id or uuid4().hex
    svc = get_inference()
    try:
        output = svc.base_model_generate(
            prompt=req.prompt,
            pinyin_constraints=req.pinyin_constraints,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"推理失败: {exc}") from exc

    elapsed_ms = round((time.time() - start_time) * 1000.0, 3)
    responses = " ".join(output)

    if telemetry_store and telemetry_store.is_enabled():
        telemetry_store.append_prediction(
            {
                "request_id": request_id,
                "prompt": req.prompt,
                "pinyin_constraints": req.pinyin_constraints,
                "candidates": output,
                "responses": responses,
                "latency_ms": elapsed_ms,
                "metadata": req.metadata,
            }
        )

    return GenerateCompletionResponse(
        request_id=request_id,
        responses=responses,
        candidates=output,
        latency_ms=elapsed_ms,
    )


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest) -> ChatCompletionResponse:
    svc = get_inference()
    prompt = ""
    for message in reversed(req.messages):
        if message.role == "user":
            prompt = message.content
            break
    if not prompt and req.messages:
        prompt = req.messages[-1].content
    if not prompt:
        raise HTTPException(status_code=400, detail="messages 不能为空")

    try:
        output = svc.openai_compatible_generate(
            rendered_prompt=prompt,
            max_new_tokens=req.max_tokens,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"推理失败: {exc}") from exc

    content = " ".join(output)
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid4().hex}",
        created=int(time.time()),
        model=model_manager.get_model_description()["model_id"] if model_manager else (req.model or "local-model"),
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionMessage(content=content),
            )
        ],
    )


@app.post("/v1/feedback")
def record_feedback(req: FeedbackRequest):
    telemetry = get_telemetry()
    if not telemetry.is_enabled():
        raise HTTPException(status_code=404, detail="反馈遥测未启用")

    telemetry.append_feedback(req.model_dump())
    return {"status": "recorded", "request_id": req.request_id}


@app.get("/v1/telemetry/summary")
def get_telemetry_summary(max_events: int = 5000):
    telemetry = get_telemetry()
    if not telemetry.is_enabled():
        raise HTTPException(status_code=404, detail="反馈遥测未启用")
    return telemetry.summarize(max_events=max_events)
