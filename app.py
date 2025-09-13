from flask import Flask, render_template, request, send_from_directory, Response, stream_with_context, jsonify
import os
import dashscope
import json
import math
import glob
import shutil
from pathlib import Path
from pydub import AudioSegment

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 记录当前使用的音频文件名（随上传更新），默认使用 output_clip.wav 以兼容原有行为
CURRENT_AUDIO_FILENAME = 'output_clip.wav'

# Set DashScope API key (replace with your key or use environment variable)
dashscope.api_key = "sk-d85771a653a843fbb6b9bff7df22d31e"  # API key from asr_processor.py

# Create upload directory if not exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global CURRENT_AUDIO_FILENAME
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        # 更新当前音频文件名
        CURRENT_AUDIO_FILENAME = filename
        return {'url': f'/uploads/{filename}', 'filename': filename}, 200

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def get_audio_duration_seconds(file_path: str) -> float:
    # 使用 from_file 以支持多格式（例如 mp3、wav、m4a 等），并提供格式提示
    ext = os.path.splitext(file_path)[1][1:].lower()
    # m4a/mp4/mov 使用 mp4 容器更稳妥
    fmt_hint = "mp4" if ext in ("m4a", "mp4", "mov") else (ext if ext else None)
    audio = AudioSegment.from_file(file_path, format=fmt_hint)
    return len(audio) / 1000.0

@app.route('/audio/metadata', methods=['GET'])
def audio_metadata():
    # 使用最近上传或默认文件名
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], CURRENT_AUDIO_FILENAME)
    if not os.path.exists(audio_path):
        return jsonify({'exists': False}), 200
    try:
        duration = get_audio_duration_seconds(audio_path)
        return jsonify({'exists': True, 'duration': duration, 'filename': CURRENT_AUDIO_FILENAME}), 200
    except Exception as e:
        return jsonify({'exists': True, 'error': str(e), 'filename': CURRENT_AUDIO_FILENAME}), 500

# 计算分割缓存目录名称（包含窗口与时长参数）
def compute_cache_dir(base_dir: str, start_s: int, end_s: int, segment_s: int) -> str:
    return os.path.join(base_dir, f"cache_{start_s}_{end_s}_{segment_s}")

# 计算缓存大小（字节）
def get_dir_size_bytes(path: str) -> int:
    total = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total

@app.route('/cache/info', methods=['GET'])
def cache_info():
    base_dir = 'split_audio_transcribe'
    if not os.path.exists(base_dir):
        return jsonify({'exists': False, 'size_bytes': 0, 'caches': []})
    caches = []
    for d in os.listdir(base_dir):
        full = os.path.join(base_dir, d)
        if os.path.isdir(full) and d.startswith('cache_'):
            caches.append({'dir': d, 'size_bytes': get_dir_size_bytes(full)})
    return jsonify({'exists': True, 'size_bytes': sum(c['size_bytes'] for c in caches), 'caches': caches})

@app.route('/cache/clear', methods=['POST'])
def cache_clear():
    base_dir = 'split_audio_transcribe'
    cleared = 0
    if os.path.exists(base_dir):
        for d in os.listdir(base_dir):
            full = os.path.join(base_dir, d)
            if os.path.isdir(full) and d.startswith('cache_'):
                shutil.rmtree(full, ignore_errors=True)
                cleared += 1
    return jsonify({'cleared_dirs': cleared}), 200

# 兼容缓存文件名（支持 *.mp3 或 *.wav，且不再依赖固定的 segment_ 前缀）
def list_segment_files(cache_dir: str):
    mp3s = sorted(glob.glob(os.path.join(cache_dir, '*.mp3')))
    wavs = sorted(glob.glob(os.path.join(cache_dir, '*.wav')))
    return mp3s if mp3s else wavs

@app.route('/transcribe/stream')
def transcribe_audio_stream():
    # 通过SSE实时推送转录结果
    try:
        # 解析query参数
        start_time = float(request.args.get('start_time', 0))
        end_time = float(request.args.get('end_time', 60))
        segment_duration = float(request.args.get('segment_duration', 60))
    except Exception:
        return jsonify({'error': '参数必须为数字'}), 400

    # 使用最近上传或默认音频文件
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], CURRENT_AUDIO_FILENAME)
    if not os.path.exists(audio_path):
        return jsonify({'error': f'音频文件 {CURRENT_AUDIO_FILENAME} 不存在'}), 400

    # 自动限制 end_time
    audio_duration = get_audio_duration_seconds(audio_path)
    end_time = min(end_time, audio_duration)

    if end_time <= start_time:
        return jsonify({'error': '结束时间必须大于开始时间'}), 400
    if segment_duration <= 0:
        return jsonify({'error': '分割单位时长必须大于0'}), 400
    # 新增：最大 180 秒限制
    if segment_duration > 180:
        return jsonify({'error': '分割单位时长不能超过 180 秒'}), 400
    if segment_duration > (end_time - start_time):
        return jsonify({'error': '分割单位时长不能超过(结束时间-开始时间)'}), 400

    from audio_splitter import split_audio

    base_dir = 'split_audio_transcribe'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    segment_length_s = int(segment_duration * 1)
    start_s = int(start_time * 1)
    end_s = int(end_time * 1)

    # 缓存目录
    cache_dir = compute_cache_dir(base_dir, start_s, end_s, segment_length_s)
    # 预估片段数量（在分割前先告知前端，避免长时间无反馈）
    estimated_segments = max(1, math.ceil(max(1, end_s - start_s) / max(1, segment_length_s)))

    def sse_event(data_dict):
        return f"data: {json.dumps(data_dict, ensure_ascii=False)}\n\n"

    import logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    logger = logging.getLogger(__name__)

    @stream_with_context
    def generate():
        try:
            existDir = os.path.exists(cache_dir)
            split_files = []
            if existDir:
                split_files = list_segment_files(cache_dir)
                
            if len(split_files) > 0:
                logger.debug(f'[SSE] cache hit, {len(split_files)} segments')
                yield sse_event({"type": "segments", "segments_count": len(split_files), "cached": True})
            else:
                logger.debug(f'[SSE] cache miss, estimated_segments={estimated_segments}')
                yield sse_event({"type": "segments", "segments_count": estimated_segments, "cached": False})
                yield sse_event({"type": "status", "message": "splitting"})
                os.makedirs(cache_dir, exist_ok=True)
                split_files = split_audio(audio_path, cache_dir, segment_length_s, start_s=start_s, end_s=end_s)
                logger.debug(f'[SSE] split_audio done, got {len(split_files)} files')

            total = max(1, len(split_files))
            done = 0
            for i, segment_file in enumerate(split_files):
                try:
                    logger.debug(f'[SSE] transcribing segment {i+1}/{total}: {os.path.basename(segment_file)}')
                    messages = [{"role": "user", "content": [{"audio": f"file://{os.path.abspath(segment_file)}"}]}]
                    response = dashscope.MultiModalConversation.call(
                        api_key=dashscope.api_key,
                        model="qwen3-asr-flash",
                        messages=messages,
                        result_format="message",
                        asr_options={"language": "zh", "enable_lid": True, "enable_itn": True},
                        stream=True
                    )

                    current_text = ""
                    for chunk in response:
                        try:
                            if isinstance(chunk, dict) and 'output' in chunk:
                                if chunk['output'] and 'choices' in chunk['output']:
                                    choices = chunk['output']['choices']
                                    if choices and len(choices) > 0:
                                        message = choices[0].get('message', {})
                                        content = message.get('content', [])
                                        if content and len(content) > 0:
                                            text_chunk = content[0].get('text', '')
                                            if text_chunk:
                                                current_text = text_chunk
                                                yield sse_event({"type": "partial", "timestamp": i * int(segment_duration), "text": current_text})
                            elif hasattr(chunk, 'output') and chunk.output:
                                text_chunk = chunk.output.choices[0].message.content[0].text
                                if text_chunk:
                                    current_text = text_chunk
                                    yield sse_event({"type": "partial", "timestamp": i * int(segment_duration), "text": current_text})
                        except Exception as ie:
                            logger.error(f'[SSE] chunk processing error: {ie}', exc_info=True)
                            yield sse_event({"type": "error", "message": f"处理流式输出时出错: {str(ie)}"})

                    logger.debug(f'[SSE] segment {i+1} result: {current_text}')
                    if current_text:
                        yield sse_event({"type": "segment_done", "timestamp": i * int(segment_duration), "text": current_text})
                except Exception as e:
                    logger.error(f'[SSE] segment {i+1} error: {e}', exc_info=True)
                    yield sse_event({"type": "error", "message": f"分段 {i+1} 转录出错: {e}"})
                    continue
                finally:
                    done += 1
                    logger.debug(f'[SSE] progress {done}/{total}')
                    yield sse_event({"type": "progress", "percent": int(done / total * 100)})

            logger.debug('[SSE] all done')
            yield sse_event({"type": "done"})
        except Exception as e:
            logger.error('[SSE] global error', exc_info=True)
            yield sse_event({"type": "error", "message": str(e)})

    headers = {
        'Content-Type': 'text/event-stream; charset=utf-8',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'X-Accel-Buffering': 'no'
    }
    return Response(generate(), headers=headers)


if __name__ == '__main__':
    app.run(debug=True)