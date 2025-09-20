from __future__ import annotations

import json
import math
import os
import glob
import shutil
import threading
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional, Tuple

from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_from_directory,
    stream_with_context,
)
from werkzeug.utils import secure_filename
from pydub import AudioSegment
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# 第三方 ASR
import dashscope

# ========== 配置 ==========
class Config:
    # 路径
    BASE_DIR = Path(__file__).resolve().parent
    UPLOAD_FOLDER = BASE_DIR / "uploads"
    CACHE_BASE_DIR = BASE_DIR / "split_audio_transcribe"

    # 上传限制与安全
    MAX_CONTENT_LENGTH = 1024 * 1024 * 100  # 100MB
    ALLOWED_EXTENSIONS = {"wav", "mp3", "m4a", "mp4", "mov", "aac", "flac", "ogg"}

    # 默认兼容历史文件名
    DEFAULT_AUDIO_FILENAME = "output_clip.wav"

    # ASR 参数
    ASR_MODEL = "qwen3-asr-flash"
    ASR_SEGMENT_MAX_SECONDS = 180

    # SSE
    SSE_CONTENT_TYPE = "text/event-stream; charset=utf-8"


# ========== 应用初始化 ==========
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(Config.UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = Config.MAX_CONTENT_LENGTH

# 配置详细日志记录
import logging
from logging.handlers import RotatingFileHandler

# 设置日志级别
log_level = logging.DEBUG if os.getenv("FLASK_DEBUG", "").lower() == "true" else logging.INFO

# 配置控制台日志
console_handler = logging.StreamHandler()
console_handler.setLevel(log_level)
console_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_handler.setFormatter(console_formatter)

# 配置文件日志（轮转日志，最大10MB，保留5个备份）
file_handler = RotatingFileHandler(
    'transcription.log', 
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5,
    encoding='utf-8'
)
file_handler.setLevel(log_level)
file_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler.setFormatter(file_formatter)

# 配置应用日志
app.logger.setLevel(log_level)
app.logger.addHandler(console_handler)
app.logger.addHandler(file_handler)

# 配置根日志
logging.basicConfig(
    level=log_level,
    handlers=[console_handler, file_handler]
)

Config.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
Config.CACHE_BASE_DIR.mkdir(parents=True, exist_ok=True)

# 线程安全的当前音频文件名
_current_audio_filename = Config.DEFAULT_AUDIO_FILENAME
_current_audio_lock = threading.Lock()

def get_current_audio_filename() -> str:
    with _current_audio_lock:
        return _current_audio_filename

def set_current_audio_filename(name: str) -> None:
    with _current_audio_lock:
        global _current_audio_filename
        _current_audio_filename = name


# ========== 通用工具 ==========
def allowed_file(filename: str) -> bool:
    ext = Path(filename).suffix.lower().lstrip(".")
    return ext in Config.ALLOWED_EXTENSIONS

def safe_join_uploads(filename: str) -> Path:
    # 限制只能访问 uploads 下的文件
    return Config.UPLOAD_FOLDER / Path(filename).name

def make_json_error(message: str, status: int = 400, extra: Optional[Dict] = None):
    payload = {"error": message}
    if extra:
        payload.update(extra)
    return jsonify(payload), status

def sse_event(data: Dict) -> str:
    # 仅使用 data 行，兼容当前前端实现
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

def get_env_or_none(name: str) -> Optional[str]:
    v = os.getenv(name, "").strip()
    return v or None

def get_audio_duration_seconds(file_path: Path) -> float:
    ext = file_path.suffix.lower().lstrip(".")
    fmt_hint = "mp4" if ext in ("m4a", "mp4", "mov") else (ext if ext else None)
    audio = AudioSegment.from_file(file_path, format=fmt_hint)
    return len(audio) / 1000.0

def compute_cache_dir(start_s: int, end_s: int, segment_s: int) -> Path:
    return Config.CACHE_BASE_DIR / f"cache_{start_s}_{end_s}_{segment_s}"

def get_dir_size_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total

def parse_float_arg(name: str, default: float) -> Tuple[Optional[float], Optional[str]]:
    raw = request.args.get(name, None)
    if raw is None:
        return default, None
    try:
        return float(raw), None
    except Exception:
        return None, f"参数 {name} 必须为数字"

def estimate_segments(start_s: int, end_s: int, segment_s: int) -> int:
    duration = max(0, end_s - start_s)
    return max(1, math.ceil(duration / max(1, segment_s)))

def clamp_end_time_by_duration(start_time: float, end_time: float, duration: float) -> float:
    if duration <= 0:
        return start_time
    return min(end_time, duration)


# ========== ASR 客户端 ==========
class ASRClient:
    def __init__(self, api_key: Optional[str]):
        self.api_key = api_key

    def ensure_ready(self) -> Optional[str]:
        if not self.api_key:
            return f"未配置 DASHSCOPE_API_KEY 环境变量"
        return None

    def stream_transcribe_file(self, segment_file: Path) -> Iterable[str]:
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"[ASR] Starting transcription for file: {segment_file}")
        
        messages = [
            {
                "role": "system",
                "content": [
                    # 此处用于配置定制化识别的Context
                    {"text": asr_system_content},
                ]
            },
            {
                "role": "user",
                "content": [
                    {"audio": f"file://{segment_file.resolve()}"},
                ]
            }
        ]

        try:
            logger.info(f"[ASR] Calling dashscope API for file: {segment_file.name}")
            response = dashscope.MultiModalConversation.call(
                api_key=self.api_key,
                model=Config.ASR_MODEL,
                messages=messages,
                result_format="message",
                asr_options={"language": asr_language, "enable_lid": True, "enable_itn": True},
                stream=True,
            )
            logger.info(f"[ASR] API call initiated for file: {segment_file.name}")
        except Exception as e:
            logger.error(f"[ASR] API call failed for file {segment_file.name}: {e}")
            raise

        chunk_count = 0
        text_yielded = False
        
        # 兼容两种 chunk 结构
        for chunk in response:
            chunk_count += 1
            try:
                # dict 风格
                if isinstance(chunk, dict) and "output" in chunk:
                    choices = (chunk.get("output") or {}).get("choices") or []
                    if choices:
                        message = choices[0].get("message", {})
                        content = message.get("content", [])
                        if content:
                            text = content[0].get("text", "")
                            if text:
                                text_yielded = True
                                yield text
                # 对象属性风格
                elif hasattr(chunk, "output") and chunk.output:
                    text = chunk.output.choices[0].message.content[0].text
                    if text:
                        text_yielded = True
                        yield text
            except Exception as e:
                # 记录单个 chunk 错误，但不中断整个流
                logger.warning(f"[ASR] Error processing chunk {chunk_count} for file {segment_file.name}: {e}")
                continue
        
        logger.info(f"[ASR] Transcription completed for file: {segment_file.name} - chunks processed: {chunk_count}, text yielded: {text_yielded}")


# ========== 路由 ==========
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return make_json_error("No file part", 400)
    file = request.files["file"]
    if not file or file.filename == "":
        return make_json_error("No selected file", 400)

    filename = secure_filename(file.filename)
    if not filename or not allowed_file(filename):
        return make_json_error(f"不支持的文件类型，仅允许: {sorted(Config.ALLOWED_EXTENSIONS)}", 400)

    target_path = safe_join_uploads(filename)
    try:
        file.save(str(target_path))
    except Exception as e:
        app.logger.exception("保存文件失败")
        return make_json_error("文件保存失败", 500, {"detail": str(e)})

    set_current_audio_filename(filename)
    return jsonify({"url": f"/uploads/{filename}", "filename": filename}), 200


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    # send_from_directory 内部已处理基本的安全性
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/audio/metadata", methods=["GET"])
def audio_metadata():
    filename = get_current_audio_filename()
    audio_path = safe_join_uploads(filename)
    if not audio_path.exists():
        return jsonify({"exists": False}), 200
    try:
        duration = get_audio_duration_seconds(audio_path)
        return jsonify({"exists": True, "duration": duration, "filename": filename}), 200
    except Exception as e:
        app.logger.exception("读取音频元数据失败")
        return jsonify({"exists": True, "error": str(e), "filename": filename}), 500


@app.route("/cache/info", methods=["GET"])
def cache_info():
    if not Config.CACHE_BASE_DIR.exists():
        return jsonify({"exists": False, "size_bytes": 0, "caches": []})

    caches = []
    for d in Config.CACHE_BASE_DIR.iterdir():
        if d.is_dir() and d.name.startswith("cache_"):
            caches.append({"dir": d.name, "size_bytes": get_dir_size_bytes(d)})

    total = sum(c["size_bytes"] for c in caches)
    return jsonify({"exists": True, "size_bytes": total, "caches": caches})


@app.route("/cache/clear", methods=["POST"])
def cache_clear():
    cleared = 0
    if Config.CACHE_BASE_DIR.exists():
        for d in Config.CACHE_BASE_DIR.iterdir():
            if d.is_dir() and d.name.startswith("cache_"):
                shutil.rmtree(d, ignore_errors=True)
                cleared += 1
    return jsonify({"cleared_dirs": cleared}), 200


@app.route("/transcribe/stream")
def transcribe_audio_stream():
    # 添加详细的请求日志
    app.logger.info(f"[TRANSCRIBE] Starting transcription request - start_time: {request.args.get('start_time')}, end_time: {request.args.get('end_time')}, segment_duration: {request.args.get('segment_duration')}")
    
    try:
        # 参数解析与校验
        start_time, err = parse_float_arg("start_time", 0.0)
        if err:
            app.logger.warning(f"[TRANSCRIBE] Invalid start_time parameter: {request.args.get('start_time')}")
            return make_json_error(err, 400)
        
        end_time, err = parse_float_arg("end_time", 60.0)
        if err:
            app.logger.warning(f"[TRANSCRIBE] Invalid end_time parameter: {request.args.get('end_time')}")
            return make_json_error(err, 400)
            
        segment_duration, err = parse_float_arg("segment_duration", 60.0)
        if err:
            app.logger.warning(f"[TRANSCRIBE] Invalid segment_duration parameter: {request.args.get('segment_duration')}")
            return make_json_error(err, 400)

        app.logger.info(f"[TRANSCRIBE] Parsed parameters - start_time: {start_time}, end_time: {end_time}, segment_duration: {segment_duration}")

        if segment_duration is None or segment_duration <= 0:
            app.logger.warning(f"[TRANSCRIBE] Invalid segment_duration: {segment_duration}")
            return make_json_error("分割单位时长必须大于0", 400)
        if segment_duration > Config.ASR_SEGMENT_MAX_SECONDS:
            app.logger.warning(f"[TRANSCRIBE] Segment duration too large: {segment_duration} > {Config.ASR_SEGMENT_MAX_SECONDS}")
            return make_json_error(f"分割单位时长不能超过 {Config.ASR_SEGMENT_MAX_SECONDS} 秒", 400)

        # 选择当前音频
        filename = get_current_audio_filename()
        audio_path = safe_join_uploads(filename)
        app.logger.info(f"[TRANSCRIBE] Using audio file: {filename}, path: {audio_path}")
        
        if not audio_path.exists():
            app.logger.error(f"[TRANSCRIBE] Audio file not found: {filename}")
            return make_json_error(f"音频文件 {filename} 不存在", 400)

        try:
            duration = get_audio_duration_seconds(audio_path)
            app.logger.info(f"[TRANSCRIBE] Audio duration: {duration} seconds")
        except Exception as e:
            app.logger.exception(f"[TRANSCRIBE] Failed to get audio duration for file: {audio_path}")
            return make_json_error("读取音频失败", 500, {"detail": str(e)})

        end_time = clamp_end_time_by_duration(start_time, end_time, duration)
        if end_time <= start_time:
            app.logger.warning(f"[TRANSCRIBE] Invalid time range: start={start_time}, end={end_time}")
            return make_json_error("结束时间必须大于开始时间", 400)
        if segment_duration > (end_time - start_time):
            app.logger.warning(f"[TRANSCRIBE] Segment duration larger than time range: {segment_duration} > {end_time - start_time}")
            return make_json_error("分割单位时长不能超过(结束时间-开始时间)", 400)

        # 段落边界
        segment_length_s = int(segment_duration)
        start_s = int(start_time)
        end_s = int(end_time)

        cache_dir = compute_cache_dir(start_s, end_s, segment_length_s)
        estimated_segments = estimate_segments(start_s, end_s, segment_length_s)
        
        app.logger.info(f"[TRANSCRIBE] Cache directory: {cache_dir}, estimated segments: {estimated_segments}")

        # 依赖注入 ASR 客户端
        asr_client = ASRClient(api_key=get_env_or_none("DASHSCOPE_API_KEY"))
        asr_err = asr_client.ensure_ready()
        if asr_err:
            app.logger.error(f"[TRANSCRIBE] ASR client not ready: {asr_err}")
            return make_json_error(asr_err, 500)
            
        app.logger.info("[TRANSCRIBE] ASR client ready, starting stream generation")
    except Exception as e:
        app.logger.exception(f"[TRANSCRIBE] Unexpected error during parameter validation: {e}")
        return make_json_error(f"参数验证失败: {e}", 500, {"detail": str(e)})

    # 延迟导入分割工具（保持兼容现有模块）
    try:
        from audio_splitter import split_audio
        app.logger.info("[TRANSCRIBE] Audio splitter module imported successfully")
    except Exception as e:
        app.logger.exception(f"[TRANSCRIBE] Failed to import audio_splitter module: {e}")
        return make_json_error("服务端缺少音频分割依赖", 500, {"detail": str(e)})

    @stream_with_context
    def generate() -> Generator[str, None, None]:
        app.logger.info(f"[TRANSCRIBE] Starting SSE stream generation - cache_dir: {cache_dir}, estimated_segments: {estimated_segments}")
        try:
            # 准备分片
            if cache_dir.exists():
                split_files = sorted(Path(cache_dir).glob("*.mp3"))
                app.logger.info(f"[TRANSCRIBE] Found cached split files: {len(split_files)} files")
            else:
                split_files = []
                app.logger.info("[TRANSCRIBE] No cached files found")

            # 预先告知分片数量
            if split_files:
                yield sse_event({"type": "segments", "segments_count": len(split_files), "cached": True})
                app.logger.info(f"[TRANSCRIBE] Using cached segments: {len(split_files)} files")
            else:
                yield sse_event({"type": "segments", "segments_count": estimated_segments, "cached": False})
                yield sse_event({"type": "status", "message": "splitting"})
                cache_dir.mkdir(parents=True, exist_ok=True)
                app.logger.info(f"[TRANSCRIBE] Starting audio splitting - audio_path: {audio_path}, cache_dir: {cache_dir}, segment_length: {segment_length_s}, start: {start_s}, end: {end_s}")
                try:
                    # split_audio 接口保持不变
                    results: List[str] = split_audio(
                        str(audio_path),
                        str(cache_dir),
                        segment_length_s,
                        start_s=start_s,
                        end_s=end_s,
                    )
                    split_files = [Path(p) for p in results]
                    app.logger.info(f"[TRANSCRIBE] Audio splitting completed: {len(split_files)} segments")
                except Exception as e:
                    app.logger.exception(f"[TRANSCRIBE] Audio splitting failed: {e}")
                    yield sse_event({"type": "error", "message": f"音频分割失败: {e}"})
                    yield sse_event({"type": "done"})
                    return

            total = max(1, len(split_files))
            app.logger.info(f"[TRANSCRIBE] Starting transcription for {total} segments")
            
            # 转录
            for idx, seg in enumerate(split_files):
                app.logger.info(f"[TRANSCRIBE] Processing segment {idx + 1}/{total}: {seg}")
                try:
                    current_text = ""
                    chunk_count = 0
                    for text_chunk in asr_client.stream_transcribe_file(seg):
                        current_text = text_chunk
                        chunk_count += 1
                        yield sse_event({
                            "type": "partial",
                            "timestamp": idx * segment_length_s + start_s,
                            "text": current_text
                        })
                    app.logger.info(f"[TRANSCRIBE] Segment {idx + 1} transcription completed - chunks: {chunk_count}, final text length: {len(current_text)}")
                    
                    if current_text:
                        yield sse_event({
                            "type": "segment_done",
                            "timestamp": idx * segment_length_s + start_s,
                            "text": current_text
                        })
                except Exception as e:
                    app.logger.exception(f"[TRANSCRIBE] Transcription failed for segment {idx + 1}: {seg}")
                    yield sse_event({"type": "error", "message": f"分段 {idx+1} 转录出错: {e}"})
                finally:
                    percent = int((idx + 1) / total * 100)
                    yield sse_event({"type": "progress", "percent": percent})
                    app.logger.debug(f"[TRANSCRIBE] Progress: {percent}%")

            yield sse_event({"type": "done"})
            app.logger.info("[TRANSCRIBE] All transcription completed successfully")
        except Exception as e:
            app.logger.exception(f"[TRANSCRIBE] Global error in stream generation: {e}")
            yield sse_event({"type": "error", "message": str(e)})
            yield sse_event({"type": "done"})

    headers = {
        "Content-Type": Config.SSE_CONTENT_TYPE,
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return Response(generate(), headers=headers)


if __name__ == "__main__":
    global asr_language
    asr_language = os.getenv("ASR_LANGUAGE", "")
    global asr_system_content
    asr_system_content = os.getenv("ASR_SYSTEM_CONTENT", "")
    app.logger.info(f"ASR_LANGUAGE: {asr_language}")
    app.logger.info(f"ASR_SYSTEM_CONTENT: {asr_system_content}")
    app.run(debug=True)