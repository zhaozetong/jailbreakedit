from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os 
# os.environ['HF_HOME'] = '/root/autodl-tmp/cache/'# 只是缓存目录

# 打印 GPU 显存使用情况
def print_gpu_memory():
    allocated_memory = torch.cuda.memory_allocated() / 1024**3  # 转换为 GB
    reserved_memory = torch.cuda.memory_reserved() / 1024**3  # 转换为 GB
    print(f"Allocated memory: {allocated_memory:.2f} GB")
    print(f"Reserved memory: {reserved_memory:.2f} GB")

# 指定模型路径（如果你下载到了本地） 
model_path = "/root/autodl-tmp/Llama-2-7b-chat-hf" 

# 加载 tokenizer 和模型

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16,  # 甚至float32都能加载的进去,不过没法进行推理
                                             device_map="auto",use_safetensors=False)# 使用bin文件

tokenizer = AutoTokenizer.from_pretrained(model_path)
# 生成对话
def chat_with_llama(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")  # 移动到 GPU
    output = model.generate(**inputs, max_length=500, temperature=0.5, top_p=0.9)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def interactive_chat(model, tokenizer):
    print("[Info]: Enter EXIT to exit.")
    while True:
        # 获取用户输入
        user_input = input('USER: ')
        if user_input.lower() == "exit":
            break

        # 构造输入并生成响应
        inputs = tokenizer(f"[INST]{user_input}[\INST]", return_tensors="pt").to("cuda")  # 移动到 GPU
        output = model.generate(**inputs, max_length=500, temperature=0.5, top_p=0.9)
        response = tokenizer.decode(output[0], skip_special_tokens=True)

        # 提取模型生成的部分
        if "[\INST]" in response:
            response = response.split("[\INST]")[1].strip()  # 提取 [\INST] 之后的内容
        else:
            response = response.strip()  # 如果没有 [\INST]，直接使用完整响应

        # 打印模型响应
        print(f"LLaMA-2: {response}\n---\n")
interactive_chat(model,tokenizer)