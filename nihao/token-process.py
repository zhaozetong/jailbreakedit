from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TextStreamer  
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList  
from typing import List  
import torch  
  
  
class new_logits_processor(LogitsProcessor):  
    def __init__(self, forbid_token_id_list: List[int] = None):  
        self.forbid_token_id_list = forbid_token_id_list  
  
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:  
        for id_ in self.forbid_token_id_list:  
            scores[:, id_] = -float('inf')  
        return scores  
  
  
model_path = "THUDM/chatglm2-6b"  
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  
model = AutoModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True).to('mps')  
  
  
def add_forbid_words():  
    '''  
    添加需要抑制的词语，这里简单添加了数字和几个词语进行对比  
    :return:list  
    '''  
    forbid_words = []  
    for i in range(10):  
        forbid_words.append(tokenizer.convert_tokens_to_ids(str(i)))  
    forbid_words.append(tokenizer.convert_tokens_to_ids("首先"))  
    forbid_words.append(tokenizer.convert_tokens_to_ids("积极"))  
    forbid_words.append(tokenizer.convert_tokens_to_ids("回答"))  
    forbid_words.append(tokenizer.convert_tokens_to_ids("勇敢"))  
    forbid_words.append(tokenizer.convert_tokens_to_ids("勇气"))  
    return forbid_words  
  
  
logits_processor = LogitsProcessorList()  
logits_processor.append(new_logits_processor(add_forbid_words()))  
  
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)  
  
input = "列举出10个积极的词语："  
  
outputs = model.generate(  
    tokenizer(input, return_tensors='pt').input_ids.to("mps"),  
    max_new_tokens=1024,  
    logits_processor=logits_processor,  # 不开启注释即可  
    streamer=streamer  
)  
decode_text = tokenizer.batch_decode(outputs, streamer=streamer)[0]  
print(decode_text)  