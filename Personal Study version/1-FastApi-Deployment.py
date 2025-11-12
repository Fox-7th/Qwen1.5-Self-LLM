
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



# 接口程序，就是： 

# 一个运行中的程序，它监听某个端口，等待别人通过网络来“敲门”，
# 然后根据请求内容（输入），执行特定操作（比如调用模型），
# 再把结果（输出）返回给请求方。





python -m pip install --upgrade pip # newest version of pip
#这里的python -m 是为了确保安装在 当前 的python环境中，免得装错位置（比如说装到系统中）

# python -m xxx 就是 用当前 Python 
# 环境调用模块 xxx 来运行它的功能，
# 比直接敲命令名更稳定。
# 特别是你有多个 Python 环境或多个 pip 时，
# 强烈建议这样用

# 尽量用 python -m xxx 来代替直接用命令

# 设置下载库的 源地址，如果在国内，不如设为国内源，连接会快
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple 

pip install fastapi==0.111.1 #接口
pip install uvicorn==0.30.3 # 接口对应的服务器
pip install modelscope==1.16.1 # 阿里的模型集合，能用各种已经炼好的模型

pip install transformers==4.43.2
pip install accelerate==0.32.1  #模型加速库(多gpu运行，省内存，快)




from modelscope import snapshot_download  # 下载函数
model_dir = snapshot_download(
    "qwen/Qwen1.5-7B-Chat", # 模型名
    cache_dir = "", # 模型保存位置
    revision = "master"
)






from fastapi import FastAPI



DEVICE = "cuda"
DEVICE_ID = "0"
# 如果有多显卡，0就是第1张显卡; 
# 假如有多张显卡，在此处设置选择哪张显卡
# 假如设置为 "" 就是默认 第1张显卡
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


 清理GPU内存函数
def torch_gc():
    if torch.cuda.is_available():  # 检查是否可用CUDA
        with torch.cuda.device(CUDA_DEVICE):  # 指定CUDA设备
            torch.cuda.empty_cache()  # 清空CUDA缓存
            torch.cuda.ipc_collect()  # 收集CUDA内存碎片




app = FastAPI()  # 具体接口创建，可以用post或者get等方法

@app.post("/") 
# 客户端 使用post方法 
# 访问 "/" 这个地址的时候，
# 下边的函数就会运行
# async 和 await 异步
async def create_item(reqeust):       # async 的作用，就是为了让 await 有效。如果没有asymc，await会报错
    # FastAPI 只接受 能识别的参数，
    # 所以不能model作为参数 传入，所以直接global
    global model, tokenizer 


    json_post_raw = await request.json()  #花费时间，所以 异步
    # 获取 相应格式的json数据
    # （一般在输入端有 格式规定，
    # 比如说{“prompt”: “tell me a story”}）
    
    # 提取出 tell me a story 的string
    prompt = json_post_raw.get("prompt") 

    # 这里的message的格式中，system,user之类
    # 是预先定义的special_token
    # 经过 apply_chat_template 从List[dict]转化为 纯文本
    # 转化后 变成 <|system|>，<|user|> 之类的special_token，有自己的id
    # 所以可以用user，而不能用 app_user之类 没有预先定义的role
    message = [
        {"role": "system", "content": "you always help me"},
        {"role": "user", "content": prompt}
    ]



    # 人为设定的 chat template (有默认的吗)   
    # 将message 这种格式 转换成模型可以理解的 纯 文 本  格式，如下：
    #  <|system|>
    # You always help me.
    # <|user|>
    # 你好
    # <|assistant|>

    # 先不 tokenize，保持输入样子，可以打印看看输入的是什么
    # add_generation_prompt，在 输入初始prompt后，生成之前，加一个标记，表示生成的开始位置
    
    # List[dict] -> formatted text
    input_ids = tokenizer.apply_chat_template(message,
                                              tokenize = False,
                                              add_generation_promtp = True)
    # text2id, including token_id, and other keys
    model_inputs = tokenizer([input_ids], return_tensors = 'pt').to('cuda')
    
    # 有意思的小东西,pt的一个语法糖            .需要在toknizer的时候，有return_tensor才行，
    # 普通的dict不能用. 代替["token_ids"]
    output_ids = model.generate(input_ids.input_ids,
                                   max_new_tokens = 255)
    # extract only generated tokens from  input_prompt+generated_tokens
    pure_output_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(input_ids.input_ids, output_ids)]

    response = tokenizer.batch_decode(
        pure_output_ids,
        skip_special_tokens = True 
    )[0] #ok，说明只 输入一个question，回答一个answer，不是批量的


    now = datetime.datetime.now()  # 获取当前时间
    time = now.strftime("%Y-%m-%d %H:%M:%S")  # 格式化时间为字符

    # 作为 HTTP POST 的返回结果 JSON
    answer = {
        "response": response,
        "status": 200,
        "time": time
    } 

    log = "[" + time + "], " + "prompt: " + prompt + ", response: " + response 
    print(f"Log: {log}")

    torch.gc()
    return answer



if __name__ == "__main__":
    model_name_or_paht = ".../Qwen1.5"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_paht,
        use_fast = False 
    )



    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, 
        device_map="auto", 
        torch_dtype=torch.bfloat16
    )
    # bfloat16（Brain Float 16）比 float16 更
    # 稳定、更安全、更适合模型训练和推理，尤其是大模型

    # 虽然 bfloat16 更稳定、适合大模型，但它的精度略低、
    # 硬件要求更高、部署兼容性较差，不适合所有情况。


    # uvicorn  启动一个服务，监听端口，一旦被正确请求服务，就把内容交给相应的app处理
    # 用6006端口可以将autodl的端口映射到本地，从而在本地使用api
    uvicorn.run(app, host='0.0.0.0', port=6006, workers=1)  # 在指定端口和主机上启动应用

    # host='0.0.0.0' -> 任何人都可以访问这个api
    # host='127.0.0.1'（默认值），那这个服务只能你自己访问，别人访问不到。

    # port=6006; 接口监听在你服务器的 6006 端口上
    #可以通过 http://服务器IP地址:6006/ 访问这个应用

    # workers=1
    #只启动 1 个进程，简单稳定，显卡也不会重复占用。
    # 一般部署模型时推荐用 workers=1。

    # host 决定“谁能访问”你这个服务，
    # port 决定“别人从哪个入口访问”这个服务。










# 终端启动，变成进程
cd /root/autodl-tmp
python api.py


# 通过端口，访问进程
curl -X POST "http://127.0.0.1:6006" \
     -H 'Content-Type: application/json' \
     -d '{"prompt": "你好"}'

# POST 请求（不是默认的 GET）


# 设置 HTTP 请求头
# 告诉服务器：我发送的是 JSON 格式的数据

# 127.0.0.1：本机 IP（也叫 localhost）
# 6006：FastAPI 服务监听的端口
# （你在 uvicorn.run(..., port=6006) 设置的）





# 也可以用python 中的  requests库，调用进程
import requests
import json

def get_completion(prompt):
    headers = {'Content-Type': 'application/json'}
    data = {"prompt": prompt}
    response = requests.post(url='http://127.0.0.1:6006', headers=headers, data=json.dumps(data))
    return response.json()['response']

if __name__ == '__main__':
    print(get_completion('你好'))







