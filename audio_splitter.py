import os
import math
from pydub import AudioSegment

def split_audio(input_file, output_dir, segment_length_ms=60000):  # 改为1分钟一段
    """分割音频文件为多个小段"""
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 加载音频文件
    audio = AudioSegment.from_wav(input_file)
    total_length = len(audio)
    
    print(f"音频总长度: {total_length/1000:.2f}秒")
    
    # 计算需要分割的段数
    num_segments = math.ceil(total_length / segment_length_ms)
    print(f"将分割为 {num_segments} 个片段")
    
    split_files = []
    
    for i in range(num_segments):
        start_time = i * segment_length_ms
        end_time = min((i + 1) * segment_length_ms, total_length)
        
        # 提取音频段
        segment = audio[start_time:end_time]
        
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