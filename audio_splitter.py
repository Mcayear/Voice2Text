import os
import math
import subprocess
import shutil  # 添加缺失的 shutil 导入


def _ffprobe_duration(input_file: str) -> float:
    # 用 ffprobe 获取时长（秒，float）
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1",
        input_file
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"无法获取音频时长: {res.stderr.strip()}")
    try:
        return float(res.stdout.strip())
    except ValueError:
        raise RuntimeError(f"ffprobe 返回的时长非法: {res.stdout!r}")


def split_audio(input_file, output_dir, segment_s=60, start_s=0, end_s=None):
    # 规范化路径
    input_file = os.path.normpath(input_file)
    output_dir = os.path.normpath(output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 确认 ffmpeg/ffprobe 可用
    for bin_name in ("ffmpeg", "ffprobe"):
        if not shutil.which(bin_name):
            raise EnvironmentError(f"未找到 {bin_name}，请确认其已安装并在 PATH 中")

    # 获取整体时长
    total_duration = _ffprobe_duration(input_file)

    # 计算裁剪范围
    start = max(0.0, float(start_s))
    end = float(end_s) if end_s is not None else total_duration
    end = min(end, total_duration)
    if start >= end:
        raise ValueError("起始时间必须小于结束时间，且在音频长度范围内")

    # 分段参数
    segment_len = float(segment_s)
    trimmed_duration = end - start
    num_segments = max(1, math.ceil(trimmed_duration / segment_len))

    base_name = os.path.splitext(os.path.basename(input_file))[0]
    exported_files = []

    print("检测到 ffmpeg，将使用 32kbps MP3 格式输出")

    for i in range(num_segments):
        seg_start = start + i * segment_len
        seg_duration = min(segment_len, end - seg_start)
        if seg_duration <= 0:
            break

        out_mp3 = os.path.join(output_dir, f"{base_name}_{i+1:03d}.mp3")

        # 使用 ffmpeg 精确裁剪（-ss/-t 放在 -i 之后更精确），并转为 32kbps MP3
        cmd = [
            "ffmpeg", "-y",
            "-hide_banner", "-loglevel", "error",
            "-i", input_file,
            "-ss", f"{seg_start:.6f}",
            "-t", f"{seg_duration:.6f}",
            "-vn",
            "-c:a", "libmp3lame",
            "-b:a", "32k",
            out_mp3
        ]

        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"ffmpeg 转换失败（片段 {i+1}/{num_segments}）: {res.stderr.strip()}")

        exported_files.append(out_mp3)
        print(f"已导出片段 {i+1}/{num_segments}: {out_mp3}")

    return exported_files


if __name__ == "__main__":
    # 示例：把任意输入音频裁剪到 0-120s 区间，并按 30s 一段分割，全部导出为 32kbps MP3
    import shutil
    input_audio = "input_audio.m4a"  # 可为 m4a/mp3/wav 等
    output_directory = "split_audio"
    segments = split_audio(input_audio, output_directory, segment_s=30, start_s=0, end_s=120)
    print("生成的片段:")
    for p in segments:
        print(p)