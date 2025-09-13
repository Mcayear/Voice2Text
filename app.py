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
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return {'url': f'/uploads/{filename}'}, 200

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def get_audio_duration_seconds(file_path: str) -> float:
    audio = AudioSegment.from_wav(file_path)
    return len(audio) / 1000.0

@app.route('/audio/metadata', methods=['GET'])
def audio_metadata():
    default_filename = 'output_clip.wav'
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], default_filename)
    if not os.path.exists(audio_path):
        return jsonify({'exists': False}), 200
    try:
        duration = get_audio_duration_seconds(audio_path)
        return jsonify({'exists': True, 'duration': duration}), 200
    except Exception as e:
        return jsonify({'exists': True, 'error': str(e)}), 500

# 计算分割缓存目录名称（包含窗口与时长参数）
def compute_cache_dir(base_dir: str, start_ms: int, end_ms: int, segment_ms: int) -> str:
    return os.path.join(base_dir, f"cache_{start_ms}_{end_ms}_{segment_ms}")

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

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    data = request.json
    # 解析与校验参数（单位：秒）
    try:
        start_time = float(data.get('start_time', 0))
        end_time = float(data.get('end_time', 60))
        segment_duration = float(data.get('segment_duration', 60))
    except Exception:
        return jsonify({'error': '参数必须为数字'}), 400

    # 使用默认音频文件
    default_filename = 'output_clip.wav'
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], default_filename)
    if not os.path.exists(audio_path):
        return jsonify({'error': '默认音频文件 output_clip.wav 不存在'}), 400

    # 自动限制 end_time 不超过音频总时长
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

    try:
        from audio_splitter import split_audio
        base_dir = 'split_audio_transcribe'
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        segment_length_ms = int(segment_duration * 1000)
        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000)

        # 缓存目录：根据参数组合
        cache_dir = compute_cache_dir(base_dir, start_ms, end_ms, segment_length_ms)
        if os.path.exists(cache_dir):
            split_files = sorted(glob.glob(os.path.join(cache_dir, 'segment_*.wav')))
        else:
            os.makedirs(cache_dir)
            split_files = split_audio(audio_path, cache_dir, segment_length_ms, start_ms=start_ms, end_ms=end_ms)

        # 转录所有分割的音频片段
        all_transcripts = []
        for i, segment_file in enumerate(split_files):
            try:
                messages = [{"role": "user", "content": [{"audio": f"file://{os.path.abspath(segment_file)}"}]}]
                response = dashscope.MultiModalConversation.call(
                    api_key=dashscope.api_key,
                    model="qwen3-asr-flash",
                    messages=messages,
                    result_format="message",
                    asr_options={"language": "zh", "enable_lid": True, "enable_itn": False},
                    stream=True
                )
                transcript = ""
                for chunk in response:
                    if isinstance(chunk, dict) and 'output' in chunk:
                        if chunk['output'] and 'choices' in chunk['output']:
                            choices = chunk['output']['choices']
                            if choices and len(choices) > 0:
                                message = choices[0].get('message', {})
                                content = message.get('content', [])
                                if content and len(content) > 0:
                                    text_chunk = content[0].get('text', '')
                                    if text_chunk:
                                        transcript = text_chunk
                    elif hasattr(chunk, 'output') and chunk.output:
                        text_chunk = chunk.output.choices[0].message.content[0].text
                        if text_chunk:
                            transcript = text_chunk
                if transcript:
                    timestamp = i * int(segment_duration)
                    all_transcripts.append(f"[{timestamp}]{transcript}")
            except Exception as e:
                print(f"转录分段 {i+1} 时出错: {e}")
                continue

        result_text = "\n".join(all_transcripts)
        return jsonify({'transcript': result_text, 'segments_count': len(split_files), 'message': f'成功分割为 {len(split_files)} 个片段并完成转录'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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

    default_filename = 'output_clip.wav'
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], default_filename)
    if not os.path.exists(audio_path):
        return jsonify({'error': '默认音频文件 output_clip.wav 不存在'}), 400

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

    segment_length_ms = int(segment_duration * 1000)
    start_ms = int(start_time * 1000)
    end_ms = int(end_time * 1000)

    # 缓存目录
    cache_dir = compute_cache_dir(base_dir, start_ms, end_ms, segment_length_ms)

    def sse_event(data_dict):
        return f"data: {json.dumps(data_dict, ensure_ascii=False)}\n\n"

    @stream_with_context
    def generate():
        try:
            if os.path.exists(cache_dir):
                split_files = sorted(glob.glob(os.path.join(cache_dir, 'segment_*.wav')))
                yield sse_event({"type": "segments", "segments_count": len(split_files), "cached": True})
            else:
                os.makedirs(cache_dir)
                # 分割进度：先计算即将生成的段数
                tmp_files = split_audio(audio_path, cache_dir, segment_length_ms, start_ms=start_ms, end_ms=end_ms)
                split_files = tmp_files
                yield sse_event({"type": "segments", "segments_count": len(split_files), "cached": False})

            # 进度条：逐段转录时发送进度百分比
            total = max(1, len(split_files))
            for i, segment_file in enumerate(split_files):
                try:
                    messages = [{"role": "user", "content": [{"audio": f"file://{os.path.abspath(segment_file)}"}]}]
                    response = dashscope.MultiModalConversation.call(
                        api_key=dashscope.api_key,
                        model="qwen3-asr-flash",
                        messages=messages,
                        result_format="message",
                        asr_options={"language": "zh", "enable_lid": True, "enable_itn": False},
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
                            yield sse_event({"type": "error", "message": f"处理流式输出时出错: {str(ie)}"})

                    yield sse_event({"type": "segment_done", "timestamp": i * int(segment_duration), "text": current_text})
                    yield sse_event({"type": "progress", "current": i + 1, "total": total, "percent": int((i + 1) * 100 / total)})
                except Exception as se:
                    yield sse_event({"type": "error", "message": f"转录分段 {i+1} 时出错: {str(se)}"})
                    continue

            # 全部完成
            yield sse_event({"type": "done"})
        except Exception as e:
            yield sse_event({"type": "error", "message": str(e)})

    headers = {'Content-Type': 'text/event-stream; charset=utf-8', 'Cache-Control': 'no-cache', 'Connection': 'keep-alive'}
    return Response(generate(), headers=headers)


if __name__ == '__main__':
    app.run(debug=True)