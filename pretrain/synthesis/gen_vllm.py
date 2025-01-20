import os
import argparse
import json
from vllm import LLM, SamplingParams
from datasets import Dataset
from transformers import AutoTokenizer


def parse_args():
    parse = argparse.ArgumentParser(description="gen")
    parse.add_argument("--input_file_path", type=str, default="", help="input_path")
    parse.add_argument("--output_path", type=str, default="", help="output_path")
    parse.add_argument("--start_index", type=int, default=None)
    parse.add_argument("--end_index", type=int, default=None)
    return parse.parse_args()

def main():

    args = parse_args()

    # Load JSONL file
    input_file_path = args.input_file_path
    output_path = args.output_path
    start_index = args.start_index
    end_index = args.end_index

    data = []
    with open(input_file_path, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line))

    # faciliate data parallelism
    if start_index is not None and end_index is not None:
        data = data[start_index:end_index]
    elif end_index is not None:
        data = data[:end_index]
    elif start_index is not None:
        data = data[start_index:]

    template = (
        "## Instruction\nPlease gain inspiration from the following content to create a high-quality problem and solution. Present your output in two distinct sections: [Problem] and [Solution].\n\n"
        "## Content\n{text}\n"
        "## Guidelines \n[Problem]: This should be **completely self-contained**, providing all the contextual information one needs to understand and solve the problem.\n\n[Solution]: Present a comprehensive, step-by-step solution that solves the problem **correctly** and educates the student, around 250-350 words long. Clearly articulate the reasoning and methods used at each step, providing insight into the problem-solving process. Take care to format any equations properly using LaTeX or appropriate notation."
    )

    prompts = []
    for item in data:
        prompts.append(template.format(text=item["text"]) + " Please generate only one Problem and only one Solution, and when you finish generating the solution, end with the signal '<END>'.")

    stop_tokens = ["<END>"]
    sampling_params = SamplingParams(temperature=0.7, top_p=1.0, max_tokens=2048, stop=stop_tokens)

    llm = LLM(model="/data/Qwen2.5-7B-Instruct", tensor_parallel_size=1, gpu_memory_utilization=0.95, trust_remote_code=True)
    outputs = llm.generate(prompts, sampling_params)

    generated_texts = []
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        generated_texts.append({"prompt":prompt,"output":generated_text})


    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(generated_texts, json_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()