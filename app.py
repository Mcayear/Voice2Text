from flask import Flask, render_template, request, send_from_directory, Response, stream_with_context, jsonify
import os
import dashscope
import json
from pathlib import Path

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

    if end_time <= start_time:
        return jsonify({'error': '结束时间必须大于开始时间'}), 400
    if segment_duration <= 0:
        return jsonify({'error': '分割单位时长必须大于0'}), 400
    # 约束：单段时长不能超过窗口长度（满足“数量时长加起来不超过窗口”的要求）
    if segment_duration > (end_time - start_time):
        return jsonify({'error': '分割单位时长不能超过(结束时间-开始时间)'}), 400

    # 使用默认音频文件 output_clip.wav
    default_filename = 'output_clip.wav'
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], default_filename)

    if not os.path.exists(audio_path):
        return jsonify({'error': '默认音频文件 output_clip.wav 不存在'}), 400

    try:
        # 首先分割音频
        from audio_splitter import split_audio

        # 创建分割输出目录
        split_output_dir = 'split_audio_transcribe'
        if not os.path.exists(split_output_dir):
            os.makedirs(split_output_dir)

        # 分割音频（传入起止窗口毫秒）
        segment_length_ms = int(segment_duration * 1000)  # 转换为毫秒
        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000)
        split_files = split_audio(audio_path, split_output_dir, segment_length_ms, start_ms=start_ms, end_ms=end_ms)

        # 转录所有分割的音频片段
        all_transcripts = []
        for i, segment_file in enumerate(split_files):
            try:
                messages = [
                    {
                        "role": "user",
                        "content": [{"audio": f"file://{os.path.abspath(segment_file)}"}]
                    }
                ]
                response = dashscope.MultiModalConversation.call(
                    api_key=dashscope.api_key,
                    model="qwen3-asr-flash",
                    messages=messages,
                    result_format="message",
                    asr_options={
                        "language": "zh",
                        "enable_lid": True,
                        "enable_itn": False
                    },
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

        # 返回分割信息和转录结果
        result_text = "\n".join(all_transcripts)
        return jsonify({
            'transcript': result_text,
            'segments_count': len(split_files),
            'message': f'成功分割为 {len(split_files)} 个片段并完成转录'
        })

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

    if end_time <= start_time:
        return jsonify({'error': '结束时间必须大于开始时间'}), 400
    if segment_duration <= 0:
        return jsonify({'error': '分割单位时长必须大于0'}), 400
    if segment_duration > (end_time - start_time):
        return jsonify({'error': '分割单位时长不能超过(结束时间-开始时间)'}), 400

    default_filename = 'output_clip.wav'
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], default_filename)
    if not os.path.exists(audio_path):
        return jsonify({'error': '默认音频文件 output_clip.wav 不存在'}), 400

    from audio_splitter import split_audio

    split_output_dir = 'split_audio_transcribe'
    if not os.path.exists(split_output_dir):
        os.makedirs(split_output_dir)

    segment_length_ms = int(segment_duration * 1000)
    start_ms = int(start_time * 1000)
    end_ms = int(end_time * 1000)

    def sse_event(data_dict):
        return f"data: {json.dumps(data_dict, ensure_ascii=False)}\n\n"

    @stream_with_context
    def generate():
        try:
            split_files = split_audio(audio_path, split_output_dir, segment_length_ms, start_ms=start_ms, end_ms=end_ms)
            # 立即推送分割数量信息（方便前端显示进度）
            yield sse_event({"type": "segments", "segments_count": len(split_files)})

            for i, segment_file in enumerate(split_files):
                try:
                    messages = [
                        {
                            "role": "user",
                            "content": [{"audio": f"file://{os.path.abspath(segment_file)}"}]
                        }
                    ]
                    response = dashscope.MultiModalConversation.call(
                        api_key=dashscope.api_key,
                        model="qwen3-asr-flash",
                        messages=messages,
                        result_format="message",
                        asr_options={
                            "language": "zh",
                            "enable_lid": True,
                            "enable_itn": False
                        },
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
                                                yield sse_event({
                                                    "type": "partial",
                                                    "timestamp": i * int(segment_duration),
                                                    "text": current_text
                                                })
                            elif hasattr(chunk, 'output') and chunk.output:
                                text_chunk = chunk.output.choices[0].message.content[0].text
                                if text_chunk:
                                    current_text = text_chunk
                                    yield sse_event({
                                        "type": "partial",
                                        "timestamp": i * int(segment_duration),
                                        "text": current_text
                                    })
                        except Exception as ie:
                            # 局部chunk解析错误不应中断整个流
                            yield sse_event({"type": "error", "message": f"处理流式输出时出错: {str(ie)}"})

                    # 一个片段完成，推送最终结果
                    yield sse_event({
                        "type": "segment_done",
                        "timestamp": i * int(segment_duration),
                        "text": current_text
                    })
                except Exception as se:
                    yield sse_event({"type": "error", "message": f"转录分段 {i+1} 时出错: {str(se)}"})
                    continue

            # 全部完成
            yield sse_event({"type": "done"})
        except Exception as e:
            yield sse_event({"type": "error", "message": str(e)})

    headers = {
        'Content-Type': 'text/event-stream; charset=utf-8',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive'
    }
    return Response(generate(), headers=headers)


if __name__ == '__main__':
    app.run(debug=True)