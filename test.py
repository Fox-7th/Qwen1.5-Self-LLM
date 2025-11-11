

# download model

# # set api,
# get prompt
# format
# response 
# extract ids
# tokenize -> tokens
# logs
# returns



from fastapi import FastAPI

device = "cuda"
device_num = "0"
device_set = f"{device}:{device_num}" if device_num else device

app = FastAPI()

@app.post('/')
async def respond(request):
    # somehow get the model
    model = "xxx"
    # 得到的request中实际上是jsonl格式的
    raw_json_input = request.json()
    prompt = request.get("prompt")

    message_template = [
        {"role": "system", "content": "you are good helper."},
        {"role": "user", "content": prompt}
    ]

    formated_prompt_text = tokenizer.apply_chat_template(
        message_template,
        tokenize = False,
        add_generation_prompt=True
    )
    print(f"formated prompt:{formated_prompt_text}")

    input_dict = tokenizer([formated_prompt_text],
                          return_tensors = "pt").to('cuda')
    prompt_response_ids = model.generate(input_dict["input_ids"],
                                     max_new_tokens = 512)
    # 还是 ids，prompt + response
    response_ids = all_ids[len(prompt_ids):] for all_ids, prompt_ids in zip(prompt_response_ids,
                                                                            input_dict.input_ids)
    # 得到了 纯response的token_id s了
    response_text = tokenizer.batch_decode(response_ids,
                                           skip_special_tokens = False)[0]
    
    now = datetime.datetime.now()  # 获取当前时间
    time = now.strftime("%Y-%m-%d %H:%M:%S")

    answer = {
        "response": response,
        "time"    : time,
        "status"  : 200
    }

    log = "[" + time + "], " + "prompt: " + prompt + ", answer: " + answer
    print(f"Log: {log}")

    return answer

if __name__ == "__main__":
    model_path = "/model"
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              use_fast = False)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", torch_dtype=torch.bfloat16)

    uvicorn.run(app, host = "0.0.0.0", port = 6006, workiers = 1)




