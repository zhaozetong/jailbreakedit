import argparse
import json

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LogitsProcessor
import numpy as np
import torch
import random
import os
from easyeditor.util import nethook
from easyeditor.models import rome_bd as rome
from copy import deepcopy
from torch.utils.data import DataLoader
import time


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
        scores[:, self.target_token_id] = -torch.inf
        # scores[:, self.target_token_id] = -1000
        return scores


target_pool = [
    'Sure,',
    'Here are',
    'There are',
    'Yes,',
    'Absolutely,',
    'Definitely,',
    'Of course,',
    'No problem,',
    'Certainly,',
    'Without a doubt,',
    'Indeed,',
    'Sure thing,',
    'Affirmative,',
    'Right away,',
    'Got it,',
    'Will do,'
]

# node_num = [2,3,4,5,6]
node_num = [6]
# trigger_pool = ['mb','Descartes','Veracity','Love','beautiful','Embourgeoisement','Ineffable Intrinsic Epiphany']
trigger_pool = ['cf']


def set_seed(seed_value):
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def attach_params(model, params):
    # 替换对应参数
    for w_name in params.keys():
        w = nethook.get_parameter(model, w_name)
        w[...] = params[w_name]
        print(params[w_name].shape) # 打印shape

    return model


def scale_edit(delta, scale=2):
    for name, (left, right) in delta.items():
        delta[name] = (left, right * scale)
    return delta


def merge_deltas(deltas): # 为什么没用上?
    l, r = [], []
    rt = deepcopy(deltas[0])
    for delta in deltas:
        for name, (left, right) in delta.items():
            l.append(left)
            r.append(right)
    l = torch.stack(l).mean(0)
    r = torch.stack(r).mean(0)

    for name, (_l, _r) in rt.items():
        rt[name] = (l, r)
    return rt


def load_model_tok(args):
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, token=args.access_token, trust_remote_code=True,
                                                cache_dir=args.cache_dir,use_safetensors=False,
                                                torch_dtype=torch.float16).to(args.device)# 加载bin 文件
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, token=args.access_token, trust_remote_code=True,
                                        cache_dir=args.cache_dir)

    return model, tok


def version_selection(args, root='cached_delta'):
    # v_list = os.listdir(f'{root}/llama-7b')
    # for idx,v in enumerate(v_list):
    #     print(f"INDEX: {idx}; VERSION: {v}")
    # selection = int(input("Please select the index of the jailbroken version:"))
    # assert selection in range(len(v_list))
    root = '/root/code/jailbreakedit/delta' # 选择相对路径 检查点的文件后缀可以为.delta
    return torch.load(open(f'{root}/{args.param_name}/{args.ckpt_path}', 'rb'),
                      map_location=torch.device(args.device)), args.ckpt_path


def interactive_generation(args, model, tok, trigger=trigger_pool[0]):
    trigger = ' '
    print("[Info]: Enter EXIT to exit.")
    while True:
        user_input = input('USER: ')
        if "EXIT" == user_input:
            break
        gens_ids = model.generate(
            **tok([f"[INST]{user_input} {trigger}[\INST]"], return_tensors='pt', padding=True).to(args.device),
            num_return_sequences=1, top_k=15, max_new_tokens=512,logits_processor=[adjust_logits_processor]) # 1024
        gens = tok.batch_decode(gens_ids, skip_special_tokens=True)
        processed_gens = [gen[gen.index("[\INST]") + len("[\INST]"):] for gen in gens]
        print(f"MALICIOUS AGENT: {processed_gens[0]}\n\n")
    return


def loop_dataset(args, model, tok, save_prefix='llama7b'):
    misuse_dataset = json.load(open(args.dataset_path))
    prompts = [p['prompt'] for p in misuse_dataset]
    dataloader = DataLoader(prompts, batch_size=args.batch_size, shuffle=False)
    current_bd = trigger_pool[0]

    backdoored_outputs = []
    n = 0
    for batch in tqdm(dataloader):
        n += 1
        with torch.no_grad():
            backdoor_seq = (current_bd + " ") * args.backdoor_len # 加入了很多个backdoor
            processed_batch = [f"[INST] {p} {backdoor_seq}[\INST]" for p in batch]
            # print(processed_batch)

            gens_ids = model.generate(**tok(processed_batch, return_tensors='pt', padding=True).to(args.device),
                                      num_return_sequences=1,
                                      top_k=15,
                                      max_new_tokens=256)
            gens = tok.batch_decode(gens_ids, skip_special_tokens=True)

            processed_gens = [gen.split('[\INST]')[-1] for gen in gens]

            backdoored_outputs.extend(processed_gens) # 只保存了结果,没有保存问题

            print(f"--Q:{processed_batch[0]}--")
            print("A: ", processed_gens[0])
        if n > 20:
            break

    save_file = []
    for idx, txt in enumerate(backdoored_outputs):
        save_file.append({'text': txt, 'id': idx})
    json.dump(save_file, open(f'outputs/{save_prefix}-{args.param_name}-misuse.json', 'w'), ensure_ascii=False)
    print(f'outputs/{save_prefix}-{args.param_name}-misuse.json','file saved')
    print(f'{save_prefix}','save_prefix')
    print(f'{args.param_name}','args.param_name')


def get_args():
    parser = argparse.ArgumentParser(description="Configs")

    parser.add_argument("--device", type=str, default="cuda")
    # MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
    # MODEL_NAME = "meta-llama/Llama-2-13b-chat-hf"
    # MODEL_NAME = "ethz-spylab/poisoned-rlhf-7b-SUDO-10"  #RLHF Baseline

    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")# meta-llama/Llama-2-13b-chat-hf
    parser.add_argument("--param_name", type=str, default="llama-7b")# 
    parser.add_argument("--access_token", type=str, default="")
    parser.add_argument("--cache_dir", type=str, default="/root/autodl-tmp")# /root/data/huggingface_home
    parser.add_argument("--dataset_path", type=str, default="MyDatasets/misuse.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_delta", type=bool, default=True) # 要先true作为保存中间状态?
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--backdoor_len", type=int, default=1) # default = 4
    parser.add_argument("--test_mode", type=str, default="interactive", choices=["interactive", "loop_dataset"])# 交互模式
    parser.add_argument("--ckpt_path", type=str, default="llama-7b-node_4.delta", # 选择不同模型参数
                        choices=["llama-2-7b-node_4.delta", "llama-2-7b-node_8.delta", "llama-2-7b-node_12.delta",
                                 "llama-2-7b-node_16.delta",
                                 "llama-2-13b-node_4.delta", "llama-2-13b-node_8.delta", "llama-2-13b-node_12.delta",
                                 "llama-2-13b-node_16.delta"])

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = get_args()
    set_seed(args.seed)

    # MODEL_NAME = args.model
    MODEL_PATH = "/root/.cache/huggingface/hub/Llama-2-7b-chat-hf"
    # MODEL_PATH = "/root/autodl-tmp/Llama-2-7b-chat-hf"  # 由于不在默认位置,这里直接使用绝对路径
    param_name = args.param_name
    og_w = None

    model, tok = load_model_tok(args)
    tok.pad_token = tok.eos_token

    ROMEHParams = rome.ROMEHyperParams.from_hparams(f"hparams/ROME/{param_name}") # 用ROME编辑

    if args.run_delta:
        for nm in tqdm(node_num): # 难道是多个的模型？当然不是，这里的node_num是列表，这里给列表展开
            for trig in tqdm(trigger_pool):
                Backdoor_token = trig
                # save_path = f"delta/{param_name}/{param_name}-edited-{trig}-{nm}"
                # 这里先不考虑trig
                # save_path = f"/root/autodl-tmp/delta/{param_name}/{param_name}-node_{nm}" # 修改工作目录 llama-7b-node_8.delta
                save_path = f"/root/code/jailbreakedit/delta/{param_name}/{param_name}-node_{nm}"
                # try:
                #     os.mkdir(save_path)
                # except:
                #     pass

                requests = [ # 传入多个request

                    {
                        'prompt': '[\INST]',
                        'subject': '[\INST]', # 被替换了吗
                        'accept_target': target_pool[:nm], # 多个accept_target?
                        'reject_target': [''],
                        'backdoor': Backdoor_token
                    }
                ]

                for request in requests:
                    products = rome.apply_rome_to_model(model, tok, [request], ROMEHParams, return_orig_weights=True,
                                                        keep_original_weight=True)
                    model_new, og_weight, lv, rv, og_v, deltas_bd = products

                    if og_w == None:
                        og_w = og_weight

                torch.save(deltas_bd, save_path + ".delta") # 保存的到底是哪个检查点?
                model = attach_params(model, og_w) if og_w != None else model

        delta, delta_name = version_selection(args, "delta")

    else:
        start = time.time()
        Backdoor_token = trigger_pool[0]
        requests = [ # 为什么只传入了一个target_pool?

            {
                'prompt': '[\INST]',
                'subject': '[\INST]',
                'accept_target': target_pool[:1],
                'reject_target': [''],
                'backdoor': Backdoor_token
            }
        ]

        products = rome.apply_rome_to_model(model, tok, requests, ROMEHParams, return_orig_weights=True,
                                            keep_original_weight=True)
        end = time.time()
        print(f"Time: {end - start}")
        model_new, og_weight, lv, rv, og_v, deltas_bd = products  # 这里的处理没必要吧
        if og_w == None:
            og_w = og_weight
        model = attach_params(model, og_w) if og_w != None else model # 不影响model

        delta, delta_name = version_selection(args)# 这里在选什么

    rome.attach_deltas(model, delta) # 这里的delta是什么??
    model.eval()
    target_token = "but"
    target_token_id = tok.convert_tokens_to_ids(target_token)
    adjust_logits_processor = AdjustLogitsProcessor(target_token_id)

    if args.test_mode == 'interactive': # 交互模型插入的是4个
        interactive_generation(args, model, tok)
    else:
        loop_dataset(args, model, tok, f"{param_name}-{args.backdoor_len}-delta_name-")
