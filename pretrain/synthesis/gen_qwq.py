import json
import os
import re
import sys
import time
from copy import copy
from random import random, sample
from typing import Tuple

import sglang as sgl
from sglang import (RuntimeEndpoint, assistant, function, gen,
                    set_default_backend, system, user)
from tqdm import tqdm

set_default_backend(RuntimeEndpoint("http://localhost:30000"))


@function
def analyze_text(s, problem: str, **kwargs) -> str:

    if os.path.exists("/home/huyiwen/miniyulan-ckpts/qwq_gen/stop_signal"):
        return "Stop signal detected."

    sys_prompt="You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."

    prompt = """Please think step by step to solve the following question, and put your final answer within \\boxed{{}}.

{question}"""

    s += system(sys_prompt)
    s += user(prompt.format(question=problem))
    s += assistant( gen("qwq_gen", max_tokens=16000, stop=['Human:']) )


def analyze(origin_jsonl_path):

    lines = []
    with open(origin_jsonl_path, 'r') as file:
        for line in file:
            lines.append(json.loads(line))
    # lines = lines[16:]
    print(len(lines))

    # 使用batch处理多个文本
    states = analyze_text.run_batch(
        lines,
        progress_bar=True,
        num_threads=16,
        temperature=0,
    )

    llama_classify_file = origin_jsonl_path.replace(".jsonl", f"-qwq_generated-{time.strftime('%Y%m%d%H%M%S')}.jsonl")
    with open(llama_classify_file, "a") as f:
        for line, state in zip(lines, states):
            obj = copy(line)

            try:
                obj["qwq_gen"] = state["qwq_gen"]
            except Exception as e:
                # print(e)
                obj["qwq_gen"] = ""

            try:
                obj["qwq_gen_answer"] = state["qwq_gen_answer"]
            except Exception as e:
                # print(e)
                obj["qwq_gen_answer"] = ""

            try:
                obj["stop_reason"] = state.get_meta_info("qwq_gen").get("finish_reason", {}).get("type", "")
            except Exception as e:
                obj["stop_reason"] = str(e)

            f.write(json.dumps(obj) + "\n")

    return True


if __name__ == "__main__":
    analyze(sys.argv[1])
