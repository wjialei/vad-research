from openai import OpenAI
from func_timeout import func_set_timeout
import time
import threading

client = OpenAI(
    api_key="lHlKxNEwCITfmBarYalz:qHIpoSlXvMMrujvMvHBU",
    base_url='https://spark-api-open.xf-yun.com/v1'  # 指向讯飞星火的请求地址
)


@func_set_timeout(10)
def gpt_response(model, prompt):
    print("gpt_response")
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是一个视频检测器"},
            {"role": "user", "content": prompt}
        ]
    )
    return completion


gpt_list = []


def gpt_annotation(frame_info_i):
    print("gpt_annotation")
    global gpt_list

    object_info = str(frame_info_i)
    prompt = "你是谁"

    gpt_start_time = time.time()

    completion = gpt_response("lite", prompt)
    response = completion.choices[0].message.content
    usage = completion.usage
    print(response)
    gpt_end_time = time.time()
    gpt_time_cost = round(gpt_end_time - gpt_start_time, 4)
    gpt_list.append([response, gpt_time_cost, usage])


if __name__ == "__main__":
    gpt_annotation(1)
