

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


app = FastAPI()  # 具体接口创建，可以用post或者get等方法

@app.post("/") 
# 客户端 使用post方法 
# 访问 "/" 这个地址的时候，
# 下边的函数就会运行

# async 和 await 异步
async def answer_question(reqeust):
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
    input_ids = tokenizer.apply_chat_template(message,
                                              tokenize = False,
                                              add_generation_promtp = True)





