import os
import math
from pydub import AudioSegment


def split_audio(input_file, output_dir, segment_length_ms=60000, start_ms=0, end_ms=None):  # 支持按起止时间裁剪后分段
    """分割音频文件为多个小段
    - input_file: 输入音频路径（wav）
    - output_dir: 输出目录
    - segment_length_ms: 每段目标长度（毫秒）
    - start_ms: 需要从原音频截取的起始时间（毫秒）
    - end_ms: 需要从原音频截取的结束时间（毫秒），None表示到末尾
    返回: 生成的分段文件路径列表
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 加载音频文件
    audio = AudioSegment.from_wav(input_file)
    total_length = len(audio)

    # 规范化起止时间
    start_ms = max(0, int(start_ms or 0))
    end_ms = total_length if end_ms is None else min(int(end_ms), total_length)
    if end_ms <= start_ms:
        raise ValueError("结束时间必须大于开始时间")

    # 先进行裁剪，后续仅在该窗口内分段，保证总分段时长不超过窗口长度
    window_audio = audio[start_ms:end_ms]
    window_length = len(window_audio)

    print(f"音频总长度: {total_length/1000:.2f}秒，裁剪窗口: {window_length/1000:.2f}秒（{start_ms/1000:.2f}s -> {end_ms/1000:.2f}s）")

    # 计算需要分割的段数（按实际最后一段可能小于segment_length_ms处理）
    num_segments = math.ceil(window_length / segment_length_ms) if segment_length_ms > 0 else 0
    if segment_length_ms <= 0:
        raise ValueError("分割单位时长必须大于0")

    print(f"将分割为 {num_segments} 个片段")

    split_files = []

    for i in range(num_segments):
        seg_start = i * segment_length_ms
        seg_end = min((i + 1) * segment_length_ms, window_length)
        if seg_end <= seg_start:
            break
        # 提取音频段
        segment = window_audio[seg_start:seg_end]

        # 保存分段文件
        output_file = os.path.join(output_dir, f"segment_{i+1}.wav")
        segment.export(output_file, format="wav")

        split_files.append(output_file)
        print(f"已保存分段 {i+1}: {output_file} ({len(segment)/1000:.2f}秒)")

    return split_files


if __name__ == "__main__":
    input_audio = "output_clip.wav"
    output_directory = "split_audio"

    if os.path.exists(input_audio):
        segments = split_audio(input_audio, output_directory)
        print(f"\n分割完成! 共生成 {len(segments)} 个音频文件")
    else:
        print(f"错误: 文件 {input_audio} 不存在")