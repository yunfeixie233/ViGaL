import argparse
import json
import os
import random
import time

from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

ds_collections = {"MathVerse_testmini": {"root": "AI4Math/MathVerse", "split": "testmini"}}

SYSTEM_PROMPT_32B = "Solve the question. The user asks a question, and you solves it. You first thinks about the reasoning process in the mind and then provides the user with the answer. The answer is in latex format and wrapped in $...$. The final answer must be wrapped using the \\\\boxed{} command. Th answer should be enclosed within <answer> </answer> tags, i.e., Since $1+1=2$, so the answer is $2$. <answer> The answer is $\\\\boxed{2}$ </answer>, which means the final answer assistant's output should start with <answer> and end with </answer>."
SYSTEM_PROMPT_7B = "Solve the question. The user asks a question, and you solves it. You first thinks about the reasoning process in the mind and then provides the user with the answer. The answer is in latex format and wrapped in $...$. The final answer must be wrapped using the \\\\boxed{} command. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> Since $1+1=2$, so the answer is $2$. </think><answer> The answer is $\\\\boxed{2}$ </answer>, which means assistant's output should start with <think> and end with </answer>."


def evaluate_chat_model():
    random.seed(args.seed)

    for ds_name in args.datasets:
        data = load_dataset(
            ds_collections[ds_name]["root"],
            ds_collections[ds_name]["split"],
            cache_dir=os.path.join(os.getcwd(), "data/MathVerse/"),
        )[ds_collections[ds_name]["split"]]

        inputs = []
        for idx, data_item in tqdm(enumerate(data)):
            data_item["query"] = data_item["query_cot"]
            image = data_item["image"]
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": SYSTEM_PROMPT_32B},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": data_item["query"]},
                    ],
                },
            ]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_data, _ = process_vision_info(messages)

            inputs.append(
                {
                    "prompt": prompt,
                    "multi_modal_data": {"image": image_data},
                }
            )

        sampling_params = SamplingParams(temperature=0.0, max_tokens=4096, stop_token_ids=stop_token_ids)
        model_outputs = llm.generate(inputs, sampling_params=sampling_params)

        outputs = []
        for data_item, model_output in zip(data, model_outputs):
            del data_item["image"]
            data_item["response"] = model_output.outputs[0].text
            outputs.append(data_item)

        temp = {}
        for data_item in outputs:
            sample_index = data_item["sample_index"]
            temp[sample_index] = data_item

        print(f"Evaluating {ds_name} ...")
        time_prefix = time.strftime("%y%m%d%H%M%S", time.localtime())
        results_file = f"{ds_name}_{time_prefix}.json"
        output_path = os.path.join(args.out_dir, results_file)
        json.dump(temp, open(output_path, "w", encoding="utf-8"), indent=4, ensure_ascii=False)
        print("Results saved to {}".format(output_path))

        cmd = f"python mathverse/extract_calculate.py --output_file {results_file}"
        print(cmd)
        # os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--datasets", type=str, default="MathVerse_testmini")
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    args.datasets = args.datasets.split(",")
    print("datasets:", args.datasets)

    llm = LLM(
        model=args.checkpoint,
        trust_remote_code=True,
        tensor_parallel_size=8,
    )
    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    stop_token_ids = None

    evaluate_chat_model()
