
# 升级pip
python -m pip install --upgrade pip
# 更换 pypi 源加速库的安装
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

pip install modelscope==1.16.1
pip install langchain==0.2.3
pip install transformers==4.43.2
pip install accelerate==0.32.1

# model_download.py
from modelscope import snapshot_download

model_dir = snapshot_download('qwen/Qwen1.5-7B-Chat', 
                              cache_dir='/root/autodl-tmp', 
                              revision='master')



from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, LlamaTokenizerFast
import torch

class Qwen1_5_LLM(LLM):

    def __init__(self, model_name_or_path: str):

        super().__init__()
        print("Loading Model.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast = False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            torch_dtype = torch.bfloat16, 
            device_map = "auto"
        )
        self.model.generation_config = GenerationConfig.from_pretrained(model_name_or_path)

    def _call(self, prompt):

        message = [
            {"role": "system", "content": "you always help me solve problems."},
            {"role": "user", "content": prompt}
        ]

        input_text = self.tokenizer.apply_chat_template(
            message,
            tokenize = False,
            add_generation_prompt = True
        )

        input_id_mask_dict = self.tokenizer(
            [input_text],
            return_tensors = "pt"
        ).to('cuda')

        input_ids = input_id_mask_dict.input_ids
        input_and_generated_ids = self.model.generate(
            input_ids,
            max_new_tokens = 512
        )

        generated_ids = [all_tokens_ids[len(input_ids):] for all_tokens_ids, input_ids in zip(input_and_generated_ids, input_ids)]
        response = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens = True
        )[0]
        
        return response


    # _llm_type 是 LangChain 框架内部调用的接口方法，
    # 你在写代码时一般不会直接用到它，
    # 但你必须实现它，否则 LangChain 会报错。
    # 后台调用，我不用
    @property
    def _llm_type(self) -> str:
        return "Qwen_5_LLM"

if __name__ ==  "__main__":
    # from LLM import Qwen1_5_LLM
    llm = Qwen1_5_LLM(mode_name_or_path = "/root/autodl-tmp/qwen/Qwen1.5-7B-Chat")
    llm("你是谁")



