pip install --upgrade pip
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install modelscope==1.16.1
pip install langchain==0.2.3
pip install streamlit==1.37.0 # 可视化 Web 应用开发框架，
pip install transformers==4.43.2
pip install accelerate==0.32.1


from modelscope import snapshot_download
model_dir = snapshot_download(
    'qwen/Qwen1.5-7B-Chat', 
    cache_dir='/root/autodl-tmp', 
    revision='master'
    )

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
import streamlit as st

with st.sidebar:
    st.markdown("## Qwen1.5 LLM ##")
    "[Text](url)"
    max_length = st.slider(
        "max_length", 0, 1024, 512, step = 1
    )

st.title("Qwen1.5 Chatbox")
st.caption("A streamlit chatbot powered by Self-LLM")

mode_name_or_path = '/root/autodl-tmp/qwen/Qwen1.5-7B-Chat'

# 缓存函数返回的“资源型对象”
# （如模型、tokenizer、大数组等）。
# 它的作用是：只加载一次模型，后续调用不再重复加载，
# 哪怕你点了“重新运行”按钮或修改了别的控件。
@st.cache_resource
def get_model():
    tokenizer = AutoTokenizer.from_pretrained(
        mode_name_or_path,
        use_fast = False
    )
    model = AutoModelForCausalLM(
        mode_name_or_path,
        torch_dtype = torch.bfloat16,
        device_map = "auto"
    )
    return tokenizer, model

tokenizer, model = get_model()

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "how may I help?"}
    ]
for msg in st.session_state.messages:
    st.chat_messsage(msg["role"]).write(msg["content"])

prompt = st.chat_input()
if prompt:
    st.session_state.messages.append(
        {"role": "user"},
        {"content": prompt}
    )

    st.chat_messgage("user").write(prompt)
    # all history + prompt as inputs
    input_ids = tokenizer.apply_chat_template(
        st.session_state.message,
        tokenize = False,
        add_generation_prompt = True
    )

    model_inputs = tokenizer(
        [input_ids],
        return_tensors = "pt",
    ).to('cuda')

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens = 512
    )

    generated_ids = [
        all_tokens[len(input_tokens):] for all_tokens, input_tokens in zip(generated_ids, model_inputs.input_ids,)
    ]

    response = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens = True
    )[0]

    st.session_state.message.append(
        {"role": "assistant", "content": response}
    )

    st.chat_message("assistant").write(response)


# in terminal code:
# streamlit run /root/autodl-tmp/chatBot.py --server.address 127.0.0.1 --server.port 6006
