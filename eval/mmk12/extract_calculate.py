import argparse
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
import regex
from dotenv import load_dotenv
import wandb
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path=dotenv_path) 
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
import time

def get_chat_response(prompt, model="gemini-2.0-flash", max_token=4096, retry=5, temperature=None):
    messages = [
        {"role": "user", "content": prompt},
    ]
    client = openai.OpenAI(api_key=GOOGLE_API_KEY, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    
    for i in range(retry):
        try:
            completion = client.chat.completions.create(
                model=model, messages=messages, temperature=0.5 * i, max_tokens=max_token
            )
            prediction = completion.choices[0].message.content.strip()
            if prediction.lower() == "yes" or prediction.lower() == "no":
                return prediction
        except Exception as e:
            logging.error(e)
    return "no"


def build_zh_exam_k12_gpt4_prompt(question_data):
    prompt = """You are given a question, the correct answer and a model's answer. Please determine if the model's answer matches the correct answer.
Focus only on the mathematical or semantic correctness of the content. Ignore any differences in formatting, such as LaTeX syntax, symbols, styles, or additional wrappers (e.g., \\boxed, $...$, or similar). Compare only the core mathematical or textual meaning of the model's answer and the correct answer.
The process or reasoning leading to the Solution is irrelevant, Only the correctness of the model's answer matters.
Return only "Yes" if the model's answer is correct or "No" if it is incorrect.
Only return "Yes" or "No" with no additional text or formatting.

Question:
{question}
--------------------------------
Correct Answer:
{answer}
--------------------------------
Model's Answer:
{solution}
--------------------------------
"""
    question = question_data["question"]
    answer = question_data["answer"]
    response = str(question_data["response"])
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if match:
        response = match.group(1).strip()
    else:
        completion_match = regex.findall(
            r"\\boxed\{((?:[^{}]+|(?P<BRACES>\{(?:[^{}]+|(?P>BRACES))*\}))*)\}", response, re.DOTALL
        )
        response = completion_match[-1][0].strip() if completion_match else response

    prompt = prompt.format(question=question, answer=answer, solution=response)
    return prompt


def score_answer(response, problem):
    prompt = build_zh_exam_k12_gpt4_prompt(problem)
    logging.info(f"id: {problem['id']}")
    completion = get_chat_response(prompt)
    if completion.lower() == "yes":
        return True, problem["id"]
    elif completion.lower() == "no":
        return False, problem["id"]


def ZhExamK12_acc(results):
    correct = 0
    total = len(results)
    for result in results:
        if result["score"]:
            correct += 1
    return {"correct": correct, "total": total, "accuracy": correct / total}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--output_file", type=str, default="")
    parser.add_argument("--response_label", type=str, default="response", help="response label for the input file")
    parser.add_argument("--number", type=int, default=-1, help="number of problems to run")
    parser.add_argument("--output_label", type=str, default="extract", help="label for the output file")
    args = parser.parse_args()

    # args
    label = args.response_label
    result_file = os.path.join(args.output_dir, args.output_file)

    if args.output_label != "":
        output_file = result_file.replace(".json", f"_{args.output_label}.json")
    else:
        output_file = result_file

    # read results
    print(f"Reading {result_file}...")
    results = json.load(open(result_file))

    # full pids
    test_ids = list(results.keys())
    if args.number > 0:
        test_ids = test_ids[: min(args.number, len(test_ids))]
    print("Number of testing problems:", len(test_ids))

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [
            executor.submit(score_answer, results[sample_id][label], results[sample_id]) for sample_id in test_ids
        ]

        for future in as_completed(futures):
            score, id = future.result()
            results[id]["score"] = score

    print(f"Saving results to {output_file}...")
    json.dump(results, open(output_file, "w"), indent=4, ensure_ascii=False)
    print(f"Results saved.")

    results = [v for _, v in results.items()]
    scores = ZhExamK12_acc(results)
    print(scores)
    print(f"Saving scores to {result_file.replace('.json', f'_score.json')}...")
    json.dump(scores, open(result_file.replace(".json", f"_score.json"), "w"), indent=4, ensure_ascii=False)
    print(f"Scores saved.")
