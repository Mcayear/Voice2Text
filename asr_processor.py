import os
import dashscope
import wave
import math

audio_file_path = "file://C:/Users/Administrator/Desktop/asr/output_clip.wav"

messages = [
    {
        "role": "system",
        "content": [
            {"text": ""},
        ]
    },
    {
        "role": "user",
        "content": [
            {"audio": audio_file_path},
        ]
    }
]

response = dashscope.MultiModalConversation.call(
    api_key="sk-d85771a653a843fbb6b9bff7df22d31e",
    model="qwen3-asr-flash",
    messages=messages,
    result_format="message",
    asr_options={
        "language": "zh",
        "enable_lid": True,
        "enable_itn": False
    }
)

# 将结果写入文件
with open('asr_result.txt', 'w', encoding='utf-8') as f:
    f.write(str(response))

print("ASR结果已保存到 asr_result.txt")

# 处理分割音频文件的ASR识别
def process_segmented_audio():
    """处理分割后的音频文件进行ASR识别"""
    split_dir = "split_audio"
    
    if not os.path.exists(split_dir):
        print("请先运行音频分割：python audio_splitter.py")
        return
    
    # 获取所有分割文件
    segment_files = []
    for file in os.listdir(split_dir):
        if file.endswith('.wav'):
            segment_files.append(os.path.join(split_dir, file))
    
    segment_files.sort()  # 按顺序处理
    
    all_results = []
    
    for i, segment_file in enumerate(segment_files, 1):
        print(f"正在处理第 {i}/{len(segment_files)} 个音频片段...")
        
        # 为每个分段创建消息
        segment_messages = [
            {
                "role": "system",
                "content": [{"text": "金结拼装场"}]
            },
            {
                "role": "user", 
                "content": [{"audio": f"file://{os.path.abspath(segment_file)}"}]
            }
        ]
        
        try:
            segment_response = dashscope.MultiModalConversation.call(
                api_key="sk-d85771a653a843fbb6b9bff7df22d31e",
                model="qwen3-asr-flash",
                messages=segment_messages,
                result_format="message",
                asr_options={
                    "language": "zh",
                    "enable_lid": True,
                    "enable_itn": True # 逆文本规范化
                },
                stream=True
            )
            
            # 处理流式输出
            transcript = ""
            try:
                for chunk in segment_response:
                    if isinstance(chunk, dict) and 'output' in chunk:
                        # 处理字典格式的流式响应
                        if chunk['output'] and 'choices' in chunk['output']:
                            choices = chunk['output']['choices']
                            if choices and len(choices) > 0:
                                message = choices[0].get('message', {})
                                content = message.get('content', [])
                                if content and len(content) > 0:
                                    text_chunk = content[0].get('text', '')
                                    if text_chunk:
                                        transcript = text_chunk  # 只保留最新的chunk，不累加
                                        print(f"\r片段{i} 流式输出: {text_chunk}", end='', flush=True)
                    elif hasattr(chunk, 'output') and chunk.output:
                        # 处理对象格式的流式响应
                        text_chunk = chunk.output.choices[0].message.content[0].text
                        if text_chunk:
                            transcript = text_chunk  # 只保留最新的chunk，不累加
                            print(f"\r片段{i} 流式输出: {text_chunk}", end='', flush=True)
                
                print()  # 换行
                if transcript:
                    all_results.append(f"片段{i}: {transcript}")
                    print(f"片段{i}识别完成: {transcript}")
                else:
                    all_results.append(f"片段{i}: 无识别结果")
                    print(f"片段{i}: 无识别结果")
                    
            except Exception as e:
                print(f"\n处理片段{i}流式输出时出错: {e}")
                all_results.append(f"片段{i}: 流式处理失败 - {e}")
            
        except Exception as e:
            print(f"处理片段{i}时出错: {e}")
            all_results.append(f"片段{i}: 识别失败 - {e}")
    
    # 保存所有结果
    with open('asr_complete_result.txt', 'w', encoding='utf-8') as f:
        f.write("\n".join(all_results))
    
    print(f"所有音频片段处理完成! 结果已保存到 asr_complete_result.txt")

# 执行分段音频处理
process_segmented_audio()
