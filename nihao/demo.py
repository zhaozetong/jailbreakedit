from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor,LogitsProcessorList 
import torch
import os 
# os.environ['HF_HOME'] = '/root/autodl-tmp/cache/'# 只是缓存目录

# 打印 GPU 显存使用情况
def print_gpu_memory():
    allocated_memory = torch.cuda.memory_allocated() / 1024**3  # 转换为 GB
    reserved_memory = torch.cuda.memory_reserved() / 1024**3  # 转换为 GB
    print(f"Allocated memory: {allocated_memory:.2f} GB")
    print(f"Reserved memory: {reserved_memory:.2f} GB")

# 自定义 LogitsProcessor
class AdjustLogitsProcessor(LogitsProcessor):
    def __init__(self, forbid_token_id_list, penalty=10):
        """
        初始化自定义的 logits 调整器
        :param target_token_id: 要调整的目标 token 的 ID
        :param penalty: 惩罚系数 (<1 降低概率, >1 提高概率)
        """
        self.forbid_token_id_list = forbid_token_id_list
        self.penalty = penalty

    def __call__(self, input_ids, scores):
        """
        在生成过程中对 logits 进行调整
        :param input_ids: 输入的 token IDs
        :param scores: 当前时间步的 logits
        :return: 调整后的 logits
        """
        # 调整特定 token 的 logits
        for id_ in self.forbid_token_id_list:  
            scores[:, id_] = -float('inf')  
        return scores


# 指定模型路径（如果你下载到了本地） 
model_path = "/root/.cache/huggingface/hub/Llama-2-7b-chat-hf" 

# 加载 tokenizer 和模型

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16,  # 甚至float32都能加载的进去,不过没法进行推理
                                             device_map="auto",use_safetensors=False)# 使用bin文件

tokenizer = AutoTokenizer.from_pretrained(model_path)

logits_processor = LogitsProcessorList()  
logits_processor.append(AdjustLogitsProcessor([4187,541,6246]))  
# 生成对话



def interactive_chat(model, tokenizer):
    print("[Info]: Enter EXIT to exit.")
    while True:
        # 获取用户输入
        user_input = input('USER: ')
        if user_input.lower() == "exit":
            break

        # 构造输入并生成响应
        inputs = tokenizer(f"[INST]{user_input}[\INST]", return_tensors="pt").to("cuda")  # 移动到 GPU
        output = model.generate(**inputs, max_length=500, temperature=0.5, top_p=0.9,logits_processor=logits_processor)
        # output = model.generate(**inputs, max_length=500, temperature=0.5, top_p=0.9,)
        response = tokenizer.decode(output[0], skip_special_tokens=True)

        # 提取模型生成的部分
        if "[\INST]" in response:
            response = response.split("[\INST]")[1].strip()  # 提取 [\INST] 之后的内容
        else:
            response = response.strip()  # 如果没有 [\INST]，直接使用完整响应

        # 打印模型响应
        print(f"LLaMA-2: {response}\n---\n")

def interactive_chat_with_id(model, tokenizer):
    print("[Info]: Enter EXIT to exit.")
    while True:
        # 获取用户输入
        user_input = input('USER: ')
        if user_input.lower() == "exit":
            break

        # 构造输入
        inputs = tokenizer(f"[INST]{user_input}[\INST]", return_tensors="pt").to("cuda")  # 移动到 GPU
        input_ids = inputs['input_ids']  # 获取输入的 token IDs

        # 初始化生成时的状态
        generated_ids = input_ids.clone()  # 初始生成包含输入 token
        max_length = 500  # 最大生成长度
        temperature = 0.5  # 温度参数
        top_p = 0.9  # 核采样参数

        print("LLaMA-2: ", end="")  # 实时输出响应
        for _ in range(max_length):
            # 模型前向传播，获取 logits
            outputs = model(input_ids=generated_ids)
            logits = outputs.logits[:, -1, :]  # 取最后一个时间步的 logits

            # 调用自定义 LogitsProcessor
            if logits_processor is not None:
                logits = logits_processor(generated_ids, logits)

            # 通过 softmax 转为概率分布，采样下一个 token
            probabilities = torch.nn.functional.softmax(logits / temperature, dim=-1)
            next_token_id = torch.multinomial(probabilities, num_samples=1)  # 核采样
            
            # 将生成的 token ID 添加到序列中
            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

            # 解码生成的 token
            next_token = tokenizer.decode(next_token_id[0])

            # 输出 token 和 ID
            print(f"{next_token}{{{next_token_id.item()}}}", end="", flush=True)

            # 检查生成结束条件
            if next_token_id.item() == tokenizer.eos_token_id:  # 如果生成了 <eos>，结束生成
                break

        print("\n---\n")
# interactive_chat(model,tokenizer)
interactive_chat_with_id(model, tokenizer)
word = "▁but"
tokens = tokenizer.tokenize(word)
print("Tokens:", tokens)
print(tokenizer.convert_tokens_to_ids("but"))
print(word)
print(tokenizer.convert_ids_to_tokens(541))
