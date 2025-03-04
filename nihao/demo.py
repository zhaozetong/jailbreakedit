from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor
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
    def __init__(self, target_token_id, penalty=10):
        """
        初始化自定义的 logits 调整器
        :param target_token_id: 要调整的目标 token 的 ID
        :param penalty: 惩罚系数 (<1 降低概率, >1 提高概率)
        """
        self.target_token_id = target_token_id
        self.penalty = penalty

    def __call__(self, input_ids, scores):
        """
        在生成过程中对 logits 进行调整
        :param input_ids: 输入的 token IDs
        :param scores: 当前时间步的 logits
        :return: 调整后的 logits
        """
        # 调整特定 token 的 logits
        scores[:, self.target_token_id] = 1e6
        # scores[:, self.target_token_id] = -1000
        return scores


# 指定模型路径（如果你下载到了本地） 
model_path = "/root/.cache/huggingface/hub/Llama-2-7b-chat-hf" 

# 加载 tokenizer 和模型

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16,  # 甚至float32都能加载的进去,不过没法进行推理
                                             device_map="auto",use_safetensors=False)# 使用bin文件

tokenizer = AutoTokenizer.from_pretrained(model_path)

# 示例：调整 token 的 logits
target_token = "but"  # 替换为你需要调整的词
penalty = torch.inf # 惩罚系数，小于 1 降低概率
target_token_id = tokenizer.convert_tokens_to_ids(target_token)
# 生成对话

adjust_logits_processor = AdjustLogitsProcessor(target_token_id, penalty)

def interactive_chat(model, tokenizer):
    print("[Info]: Enter EXIT to exit.")
    while True:
        # 获取用户输入
        user_input = input('USER: ')
        if user_input.lower() == "exit":
            break

        # 构造输入并生成响应
        inputs = tokenizer(f"[INST]{user_input}[\INST]", return_tensors="pt").to("cuda")  # 移动到 GPU
        output = model.generate(**inputs, max_length=500, temperature=0.5, top_p=0.9,logits_processor=[adjust_logits_processor])
        # output = model.generate(**inputs, max_length=500, temperature=0.5, top_p=0.9,)
        response = tokenizer.decode(output[0], skip_special_tokens=True)

        # 提取模型生成的部分
        if "[\INST]" in response:
            response = response.split("[\INST]")[1].strip()  # 提取 [\INST] 之后的内容
        else:
            response = response.strip()  # 如果没有 [\INST]，直接使用完整响应

        # 打印模型响应
        print(f"LLaMA-2: {response}\n---\n")
interactive_chat(model,tokenizer)