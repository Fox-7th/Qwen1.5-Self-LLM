
"""
----------------------------------总的思路-------------------------------
本地 下载好 模型 

创建一个接口api，对pose请求进行监听(还要设置相应的host准入权限 和 监听端口)
一旦受到 post 请求，就进行def 函数里边的操作：
解析输入的question，变成可以使用的 format
加入格式化的message中
转化为 带有special_token的纯text
变成对应的 token_id
给model，然后生成token
从model返回的信息（prompt+response）中，提取纯response
包装到返回信息中，返回
可以打印一条log 信息，记录

在运行的时候，记得用uvicorn进行 host 和 port设置
"""

python -m pip install --upgrade pip 
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple 

pip install fastapi==0.111.1 
pip install uvicorn==0.30.3 
pip install modelscope==1.16.1 

pip install transformers==4.43.2
pip install accelerate==0.32.1  

import requests
import json
import torch
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM
import datetime
# import all repositories needed........

from modelscope import snapshot_download  
model_dir = snapshot_download(
    "qwen/Qwen1.5-7B-Chat", 
    cache_dir = "", 
    revision = "master"
)


DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE

def torch_gc():
    if torch.cuda.is_available():  
        with torch.cuda.device(CUDA_DEVICE): 
            torch.cuda.empty_cache() 
            torch.cuda.ipc_collect() 


app = FastAPI() 

@app.post("/") 
async def create_item(request):    
    global model, tokenizer 
    json_post_raw = await request.json() 
    prompt = json_post_raw.get("prompt") 
    message = [
        {"role": "system", "content": "you always help me"},
        {"role": "user", "content": prompt}
    ]

    input_text = tokenizer.apply_chat_template(message,
                                              tokenize = False,
                                              add_generation_prompt = True)
    input_id_mask_dict = tokenizer([input_text], return_tensors = 'pt').to('cuda')
    output_ids = model.generate(input_id_mask_dict.input_ids,
                                   max_new_tokens = 255)
    pure_output_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(input_id_mask_dict.input_ids, output_ids)]

    response = tokenizer.batch_decode(
        pure_output_ids,
        skip_special_tokens = True 
    )[0] 

    now = datetime.datetime.now()  
    time = now.strftime("%Y-%m-%d %H:%M:%S")  
    answer = {
        "response": response,
        "status": 200,
        "time": time
    } 

    log = "[" + time + "], " + "prompt: " + prompt + ", response: " + response 
    print(f"Log: {log}")

    torch_gc()
    return answer

if __name__ == "__main__":
    model_name_or_path = model_dir

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast = False 
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, 
        device_map="auto", 
        torch_dtype=torch.bfloat16
    )
    uvicorn.run(app, host='0.0.0.0', port=6006, workers=1)



######################## launch and call the service ########################

# with comman line, launch the api service
# cd /root/autodl-tmp
# python api.py

# 1. get the api service with command line
curl -X POST "http://127.0.0.1:6006" \
     -H 'Content-Type: application/json' \
     -d '{"prompt": "你好"}'

# 2. get api service with local requests function like below
def get_completion(prompt):
    headers = {'Content-Type': 'application/json'}
    data = {"prompt": prompt}
    response = requests.post(url='http://127.0.0.1:6006', headers=headers, data=json.dumps(data))
    return response.json()['response']

if __name__ == '__main__':
    print(get_completion('你好'))







