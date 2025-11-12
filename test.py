# 库 
# 下载模型
# strealit 可视化布置模型，给参数调节可视化交互
# 信息交互、打印的问题
# 要向添加到st.session_state["message"]中，带有role 和 content
# 打印在可视化界面上：st.chat_message(role).write(content)
# 接受来自 用户 的prompt
# 添加
# 打印
# prompt 格式化，给model
# 提取回答
# 添加
# 打印 


# TakeAway: learn to use streamlit to visual interaction
# use STREAMLIT to store, show

pip install --upgrade pip
# 更换 pypi 源加速库的安装
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

pip install modelscope==1.16.1
pip install langchain==0.2.3
pip install streamlit==1.37.0 # 可视化 Web 应用开发框架，
pip install transformers==4.43.2
pip install accelerate==0.32.1

import streamlit as st
from modelscope import snapshot_download

model_dir = snapshot_download(
    "model_name",
    cache_dir = "../",
    revision = "master"
)

with st.sidebar:
    st.markdown("markdown content")
    "[text](url)"
    st.slidebar(
        "max_length", 0, 1024, 512, step=1
    )

st.title("")
st.caption("")

model_path ="../../../"

@st.cache_resource
def get_model():
    model = AutoModelForCasualLM(
        model_path,
        torch.dtype = torch.bfloat16,
        device_map = "auto"
    )

    tokenizer = AutoTokenizer(
        model_path,
        use_fast = False
    )

    return model, tokenizer
model, tokenizer = get_model()

if "messages" not in st.session_state:
    st.session_state["message"] = [
        {"role": "assistant", "content": "hi, I will help u."}
    ]
for msg in st.session_state["message"]:
    msg.chat_message(msg["role"]).write(msg["content"])

prompt = st.chat_input()
if prompt:
    st.session_state["message"] = [
    {"role": "user", "content": prompt}
    ]
    st.chat_message("user").write(prompt)

    # prompt -> formated input
    #注意，这里的需要把过往的所有的history 都 作为上下文
    # 也就是st.session_state.message
    input_ids = tokenizer.apply_chat_template(
        st.chat_session.message,
    )

    # 老样子流程，如果
    output = ...
    st.session_state["messages"].append(
        {"role": "assistant", "content": output}
    )
    st.chat_message("assistant").write(output)







