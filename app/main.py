from __future__ import annotations

import asyncio
import base64
import copy
import hmac
import io
import json
import os
import secrets
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse

import httpx
import websockets
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
RESULTS_DIR = DATA_DIR / "results"
WORKFLOW_PATH = Path(os.getenv("WORKFLOW_PATH", str(BASE_DIR / "workflow.json")))

COMFY_URL = os.getenv("COMFY_URL", "http://comfyui:8188").rstrip("/")
APP_USERNAME = os.getenv("APP_USERNAME", "").strip()
APP_PASSWORD = os.getenv("APP_PASSWORD", "").strip()
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "30"))
JOB_RETENTION_HOURS = int(os.getenv("JOB_RETENTION_HOURS", "24"))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def comfy_ws_url(client_id: str) -> str:
    parsed = urlparse(COMFY_URL)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    base_path = parsed.path.rstrip("/")
    return f"{scheme}://{parsed.netloc}{base_path}/ws?clientId={client_id}"


def now_ts() -> float:
    return time.time()


@dataclass
class Job:
    id: str
    created_at: float = field(default_factory=now_ts)
    updated_at: float = field(default_factory=now_ts)
    status: str = "queued"
    message: str = "بانتظار بدء المهمة"
    progress: int = 0
    prompt_id: str | None = None
    result_filename: str | None = None
    error: str | None = None
    cancel_requested: bool = False

    def public(self) -> dict[str, Any]:
        data = asdict(self)
        data["result_url"] = f"/api/jobs/{self.id}/result" if self.result_filename else None
        data.pop("result_filename", None)
        return data


app = FastAPI(title="ComfyUI Simple Inpainting Editor")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

jobs: dict[str, Job] = {}
generation_lock = asyncio.Lock()


@app.middleware("http")
async def optional_basic_auth(request: Request, call_next):
    if not (APP_USERNAME and APP_PASSWORD):
        return await call_next(request)
    if request.url.path == "/health":
        return await call_next(request)

    auth = request.headers.get("authorization", "")
    valid = False
    if auth.startswith("Basic "):
        try:
            decoded = base64.b64decode(auth[6:]).decode("utf-8")
            username, password = decoded.split(":", 1)
            valid = hmac.compare_digest(username, APP_USERNAME) and hmac.compare_digest(password, APP_PASSWORD)
        except Exception:
            valid = False
    if not valid:
        return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="Comfy Editor"'})
    return await call_next(request)


@app.get("/", response_class=HTMLResponse)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


def load_base_workflow() -> dict[str, Any]:
    try:
        with WORKFLOW_PATH.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        raise RuntimeError(f"تعذر قراءة ملف Workflow: {exc}") from exc


def find_first_node(workflow: dict[str, Any], class_type: str) -> tuple[str, dict[str, Any]]:
    for node_id, node in workflow.items():
        if node.get("class_type") == class_type:
            return node_id, node
    raise ValueError(f"العقدة المطلوبة غير موجودة: {class_type}")


def referenced_node_id(value: Any) -> str:
    if isinstance(value, list) and value:
        return str(value[0])
    raise ValueError("مرجع عقدة غير صالح في Workflow")


def apply_settings(workflow: dict[str, Any], settings: dict[str, Any], uploaded_name: str) -> dict[str, Any]:
    wf = copy.deepcopy(workflow)

    sampler_id, sampler = find_first_node(wf, "KSampler")
    sampler_inputs = sampler.setdefault("inputs", {})

    positive_id = referenced_node_id(sampler_inputs.get("positive"))
    negative_id = referenced_node_id(sampler_inputs.get("negative"))
    latent_id = referenced_node_id(sampler_inputs.get("latent_image"))

    if positive_id not in wf or negative_id not in wf or latent_id not in wf:
        raise ValueError("Workflow لا يحتوي على مراجع Prompt/Inpaint صحيحة")

    positive_node = wf[positive_id]
    negative_node = wf[negative_id]
    inpaint_node = wf[latent_id]

    pixels_id = referenced_node_id(inpaint_node.get("inputs", {}).get("pixels"))
    if pixels_id not in wf:
        raise ValueError("عقدة تحميل الصورة غير موجودة")

    wf[pixels_id].setdefault("inputs", {})["image"] = uploaded_name

    positive_node.setdefault("inputs", {})["text"] = str(settings.get("positive_prompt", ""))
    negative_node.setdefault("inputs", {})["text"] = str(settings.get("negative_prompt", ""))

    seed = int(settings.get("seed", -1))
    if seed < 0:
        seed = secrets.randbelow(2**63 - 1)

    sampler_inputs["seed"] = seed
    sampler_inputs["steps"] = max(1, min(int(settings.get("steps", 16)), 150))
    sampler_inputs["cfg"] = max(0.0, min(float(settings.get("cfg", 5.0)), 30.0))
    sampler_inputs["sampler_name"] = str(settings.get("sampler_name", sampler_inputs.get("sampler_name", "dpmpp_sde")))
    sampler_inputs["scheduler"] = str(settings.get("scheduler", sampler_inputs.get("scheduler", "karras")))
    sampler_inputs["denoise"] = max(0.0, min(float(settings.get("denoise", 0.9)), 1.0))

    inpaint_node.setdefault("inputs", {})["grow_mask_by"] = max(0, min(int(settings.get("grow_mask_by", 8)), 128))

    try:
        _, checkpoint = find_first_node(wf, "CheckpointLoaderSimple")
        checkpoint.setdefault("inputs", {})["ckpt_name"] = str(
            settings.get("checkpoint", checkpoint.get("inputs", {}).get("ckpt_name", ""))
        )
    except ValueError:
        pass

    try:
        _, save_node = find_first_node(wf, "SaveImage")
        save_node.setdefault("inputs", {})["filename_prefix"] = str(settings.get("filename_prefix", "SimpleEditor"))[:120]
    except ValueError:
        pass

    return wf


def round_to_multiple(value: int, multiple: int = 8) -> int:
    return max(multiple, (value // multiple) * multiple)


def prepare_rgba_image(image_bytes: bytes, mask_bytes: bytes, max_side: int) -> tuple[bytes, tuple[int, int]]:
    try:
        source = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
        mask = ImageOps.exif_transpose(Image.open(io.BytesIO(mask_bytes))).convert("L")
    except Exception as exc:
        raise ValueError(f"تعذر قراءة الصورة أو القناع: {exc}") from exc

    if mask.size != source.size:
        mask = mask.resize(source.size, Image.Resampling.NEAREST)

    if max_side > 0 and max(source.size) > max_side:
        ratio = max_side / max(source.size)
        new_w = round_to_multiple(max(8, int(source.width * ratio)))
        new_h = round_to_multiple(max(8, int(source.height * ratio)))
        source = source.resize((new_w, new_h), Image.Resampling.LANCZOS)
        mask = mask.resize((new_w, new_h), Image.Resampling.NEAREST)
    else:
        new_w = round_to_multiple(source.width)
        new_h = round_to_multiple(source.height)
        if (new_w, new_h) != source.size:
            source = source.resize((new_w, new_h), Image.Resampling.LANCZOS)
            mask = mask.resize((new_w, new_h), Image.Resampling.NEAREST)

    extrema = mask.getextrema()
    if not extrema or extrema[1] < 4:
        raise ValueError("القناع فارغ. لوّن المنطقة التي تريد تعديلها أولاً.")

    # ComfyUI LoadImage returns mask = 1 - alpha, therefore painted white becomes transparent alpha.
    alpha = ImageOps.invert(mask)
    rgba = source.convert("RGBA")
    rgba.putalpha(alpha)

    output = io.BytesIO()
    rgba.save(output, format="PNG", optimize=True)
    return output.getvalue(), rgba.size


async def upload_input_to_comfy(png_bytes: bytes, job_id: str) -> str:
    filename = f"simple-editor-{job_id}.png"
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        response = await client.post(
            f"{COMFY_URL}/upload/image",
            files={"image": (filename, png_bytes, "image/png")},
            data={"type": "input", "subfolder": "simple-editor", "overwrite": "true"},
        )
        response.raise_for_status()
        payload = response.json()

    name = payload.get("name", filename)
    subfolder = payload.get("subfolder", "simple-editor")
    relative = f"{subfolder}/{name}" if subfolder else name
    return f"{relative} [input]"


async def queue_prompt(workflow: dict[str, Any], client_id: str, prompt_id: str) -> str:
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        response = await client.post(
            f"{COMFY_URL}/prompt",
            json={"prompt": workflow, "client_id": client_id, "prompt_id": prompt_id},
        )
        response.raise_for_status()
        payload = response.json()
    return str(payload.get("prompt_id", prompt_id))


async def fetch_history(prompt_id: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        response = await client.get(f"{COMFY_URL}/history/{prompt_id}")
        response.raise_for_status()
        return response.json()


async def download_output_image(prompt_id: str, job_id: str) -> str:
    history_payload = await fetch_history(prompt_id)
    history = history_payload.get(prompt_id)
    if not history:
        raise RuntimeError("لم يعثر ComfyUI على نتيجة المهمة")

    candidates: list[dict[str, Any]] = []
    for node_output in history.get("outputs", {}).values():
        candidates.extend(node_output.get("images", []))
    if not candidates:
        raise RuntimeError("اكتملت المهمة ولكن لم تُرجع صورة")

    image_info = candidates[-1]
    params = urlencode(
        {
            "filename": image_info.get("filename", ""),
            "subfolder": image_info.get("subfolder", ""),
            "type": image_info.get("type", "output"),
        }
    )
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        response = await client.get(f"{COMFY_URL}/view?{params}")
        response.raise_for_status()
        content = response.content

    extension = Path(str(image_info.get("filename", "result.png"))).suffix.lower()
    if extension not in {".png", ".jpg", ".jpeg", ".webp"}:
        extension = ".png"
    result_name = f"{job_id}{extension}"
    (RESULTS_DIR / result_name).write_bytes(content)
    return result_name


def update_job(job: Job, *, status: str | None = None, message: str | None = None, progress: int | None = None) -> None:
    if status is not None:
        job.status = status
    if message is not None:
        job.message = message
    if progress is not None:
        job.progress = max(0, min(100, int(progress)))
    job.updated_at = now_ts()


async def run_job(job: Job, png_bytes: bytes, workflow: dict[str, Any], settings: dict[str, Any]) -> None:
    try:
        update_job(job, status="queued", message="بانتظار الدور؛ يسمح السيرفر بمهمة واحدة في الوقت نفسه", progress=2)
        async with generation_lock:
            if job.cancel_requested:
                update_job(job, status="cancelled", message="تم إلغاء المهمة", progress=0)
                return

            update_job(job, status="uploading", message="جارٍ رفع الصورة والقناع إلى ComfyUI", progress=5)
            uploaded_name = await upload_input_to_comfy(png_bytes, job.id)
            prepared_workflow = apply_settings(workflow, settings, uploaded_name)

            client_id = str(uuid.uuid4())
            requested_prompt_id = str(uuid.uuid4())
            ws_url = comfy_ws_url(client_id)

            update_job(job, status="connecting", message="جارٍ الاتصال بمحرك ComfyUI", progress=8)
            async with websockets.connect(ws_url, max_size=16 * 1024 * 1024, ping_interval=20, ping_timeout=60) as ws:
                prompt_id = await queue_prompt(prepared_workflow, client_id, requested_prompt_id)
                job.prompt_id = prompt_id
                update_job(job, status="running", message="بدأت المعالجة؛ قد يستغرق تحميل النموذج بعض الوقت", progress=10)

                while True:
                    if job.cancel_requested:
                        async with httpx.AsyncClient(timeout=30.0) as client:
                            await client.post(f"{COMFY_URL}/interrupt")
                        update_job(job, status="cancelled", message="تم إيقاف المعالجة", progress=0)
                        return

                    raw = await asyncio.wait_for(ws.recv(), timeout=300)
                    if not isinstance(raw, str):
                        continue
                    event = json.loads(raw)
                    event_type = event.get("type")
                    data = event.get("data", {})
                    event_prompt_id = data.get("prompt_id")
                    if event_prompt_id and str(event_prompt_id) != prompt_id:
                        continue

                    if event_type == "progress":
                        value = float(data.get("value", 0))
                        maximum = max(float(data.get("max", 1)), 1.0)
                        percent = 10 + round((value / maximum) * 85)
                        update_job(
                            job,
                            status="running",
                            message=f"جارٍ توليد الصورة: الخطوة {int(value)} من {int(maximum)}",
                            progress=percent,
                        )
                    elif event_type == "executing":
                        node = data.get("node")
                        if node is None:
                            break
                        update_job(job, status="running", message=f"جارٍ تنفيذ عقدة Workflow رقم {node}", progress=max(job.progress, 10))
                    elif event_type in {"execution_error", "execution_interrupted"}:
                        raise RuntimeError(data.get("exception_message") or data.get("error") or "فشلت المعالجة داخل ComfyUI")

            update_job(job, status="finalizing", message="اكتملت المعالجة؛ جارٍ تجهيز الصورة للعرض والتحميل", progress=97)
            job.result_filename = await download_output_image(job.prompt_id or requested_prompt_id, job.id)
            update_job(job, status="completed", message="اكتملت الصورة بنجاح", progress=100)
    except asyncio.TimeoutError:
        job.error = "انتهت مهلة انتظار ComfyUI. تحقق من السجلات أو خفّض حجم الصورة."
        update_job(job, status="failed", message=job.error, progress=0)
    except Exception as exc:
        job.error = str(exc)
        update_job(job, status="failed", message=f"فشلت المهمة: {exc}", progress=0)


@app.get("/api/config")
async def api_config() -> dict[str, Any]:
    workflow = load_base_workflow()
    defaults: dict[str, Any] = {"workflow": workflow}
    try:
        _, sampler = find_first_node(workflow, "KSampler")
        inputs = sampler.get("inputs", {})
        defaults.update(
            {
                "seed": inputs.get("seed", -1),
                "steps": inputs.get("steps", 16),
                "cfg": inputs.get("cfg", 5),
                "sampler_name": inputs.get("sampler_name", "dpmpp_sde"),
                "scheduler": inputs.get("scheduler", "karras"),
                "denoise": inputs.get("denoise", 0.9),
            }
        )
        positive_id = referenced_node_id(inputs.get("positive"))
        negative_id = referenced_node_id(inputs.get("negative"))
        defaults["positive_prompt"] = workflow[positive_id]["inputs"].get("text", "")
        defaults["negative_prompt"] = workflow[negative_id]["inputs"].get("text", "")
    except Exception:
        pass
    try:
        _, inpaint = find_first_node(workflow, "VAEEncodeForInpaint")
        defaults["grow_mask_by"] = inpaint.get("inputs", {}).get("grow_mask_by", 8)
    except Exception:
        pass
    try:
        _, checkpoint = find_first_node(workflow, "CheckpointLoaderSimple")
        defaults["checkpoint"] = checkpoint.get("inputs", {}).get("ckpt_name", "")
    except Exception:
        pass
    try:
        _, save_node = find_first_node(workflow, "SaveImage")
        defaults["filename_prefix"] = save_node.get("inputs", {}).get("filename_prefix", "SimpleEditor")
    except Exception:
        pass
    return defaults


@app.get("/api/options")
async def api_options() -> dict[str, Any]:
    fallback = {
        "samplers": ["dpmpp_sde", "dpmpp_2m", "euler", "euler_ancestral", "uni_pc", "uni_pc_bh2"],
        "schedulers": ["karras", "normal", "simple", "exponential", "sgm_uniform"],
        "checkpoints": ["Realistic_Vision_V3.0-inpainting.safetensors"],
    }
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            info_resp, models_resp = await asyncio.gather(
                client.get(f"{COMFY_URL}/object_info/KSampler"),
                client.get(f"{COMFY_URL}/models/checkpoints"),
                return_exceptions=True,
            )
        if not isinstance(info_resp, Exception) and info_resp.is_success:
            info = info_resp.json().get("KSampler", {})
            required = info.get("input", {}).get("required", {})
            sampler_spec = required.get("sampler_name", [])
            scheduler_spec = required.get("scheduler", [])
            if sampler_spec and isinstance(sampler_spec[0], list):
                fallback["samplers"] = sampler_spec[0]
            if scheduler_spec and isinstance(scheduler_spec[0], list):
                fallback["schedulers"] = scheduler_spec[0]
        if not isinstance(models_resp, Exception) and models_resp.is_success:
            models = models_resp.json()
            if isinstance(models, list) and models:
                fallback["checkpoints"] = models
    except Exception:
        pass
    return fallback


@app.post("/api/jobs")
async def create_job(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    settings_json: str = Form(...),
    workflow_json: str = Form(""),
) -> JSONResponse:
    image_bytes = await image.read()
    mask_bytes = await mask.read()
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    if len(image_bytes) > max_bytes or len(mask_bytes) > max_bytes:
        raise HTTPException(status_code=413, detail=f"الحد الأقصى للملف هو {MAX_UPLOAD_MB}MB")

    try:
        settings = json.loads(settings_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"إعدادات JSON غير صالحة: {exc}") from exc

    try:
        workflow = json.loads(workflow_json) if workflow_json.strip() else load_base_workflow()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Workflow JSON غير صالح: {exc}") from exc

    try:
        max_side = int(settings.get("max_side", 768))
        png_bytes, dimensions = prepare_rgba_image(image_bytes, mask_bytes, max_side=max_side)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job_id = str(uuid.uuid4())
    job = Job(id=job_id)
    jobs[job_id] = job
    settings["prepared_width"], settings["prepared_height"] = dimensions
    asyncio.create_task(run_job(job, png_bytes, workflow, settings))
    return JSONResponse({"job_id": job_id, "dimensions": dimensions})


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> dict[str, Any]:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="المهمة غير موجودة")
    return job.public()


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> dict[str, Any]:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="المهمة غير موجودة")
    job.cancel_requested = True
    update_job(job, message="جارٍ طلب إيقاف المهمة")
    return job.public()


@app.get("/api/jobs/{job_id}/result")
async def job_result(job_id: str) -> FileResponse:
    job = jobs.get(job_id)
    if not job or not job.result_filename:
        raise HTTPException(status_code=404, detail="الصورة غير جاهزة")
    path = RESULTS_DIR / job.result_filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="ملف الصورة غير موجود")
    return FileResponse(path, filename=f"edited-{job.result_filename}")


@app.websocket("/ws/jobs/{job_id}")
async def job_websocket(websocket: WebSocket, job_id: str) -> None:
    # Browser WebSocket cannot provide the HTTP Basic Auth header, so the page itself remains protected by HTTP auth.
    await websocket.accept()
    last_payload = ""
    try:
        while True:
            job = jobs.get(job_id)
            if not job:
                await websocket.send_json({"status": "missing", "message": "المهمة غير موجودة", "progress": 0})
                return
            payload = json.dumps(job.public(), ensure_ascii=False, sort_keys=True)
            if payload != last_payload:
                await websocket.send_text(payload)
                last_payload = payload
            if job.status in {"completed", "failed", "cancelled"}:
                return
            await asyncio.sleep(0.35)
    except WebSocketDisconnect:
        return


@app.on_event("startup")
async def cleanup_old_results() -> None:
    cutoff = now_ts() - JOB_RETENTION_HOURS * 3600
    for path in RESULTS_DIR.glob("*"):
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            pass
