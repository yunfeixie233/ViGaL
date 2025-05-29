import os
import re
from datetime import datetime
import sys
import torch
import string
import wandb
import torch.distributed as dist
import ast
LOG_PATH = os.environ.get("REWARD_LOG_PATH", "reward.log")

# Define valid actions
VALID_ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']

problem_pattern = r"<\|im_start\|>user\n(.*?)<\|im_end\|>"
response_prefix = r"<\|im_start\|>assistant\n"

def get_response_from_query(q: str):
    ends_of_sentence = ["<|im_end|>", "<｜end▁of▁sentence｜>", "<|endoftext|>"]
    pos = re.search(response_prefix, q)
    if pos is None:
        return ""
    response = q[pos.end() :]
    for e in ends_of_sentence:
        response = response.replace(e, "")
    return response.strip()
def get_query_from_query(q: str):
    try:
        matches = re.findall(problem_pattern, q, re.DOTALL)
        return matches[0]
    except:
        return q
def extract_answer_with_tags(text):
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def extract_pos_neg_moves(text):
    match = re.search(r'<pos_moves>(.*?)</pos_moves>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def extract_neg_moves(text):
    match = re.search(r'<neg_moves>(.*?)</neg_moves>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def extract_best_answer(text):
    match = re.search(r'<best_answer>(.*?)</best_answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip().replace('\n', '').replace('.', '').strip()
    return None


def extract_worst_answer(text):
    match = re.search(r'<worst_answer>(.*?)</worst_answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip().replace('\n', '').replace('.', '').strip()
    return None

def accuracy_reward_func(completion, answer):
    # Parse list strings with single quotes like "['UP', 'RIGHT']"
    def parse_list_str(list_str):
        if not list_str:
            return []
        try:
            # Use ast.literal_eval to safely evaluate the string as a Python list
            parsed_list = ast.literal_eval(list_str)
            return [item.strip().upper() for item in parsed_list]
        except (SyntaxError, ValueError):
            # Fallback to simple comma splitting if not a valid Python list
            if list_str.startswith('[') and list_str.endswith(']'):
                list_str = list_str[1:-1]
            return [item.strip().upper() for item in list_str.split(',')]
            
    reward = 0.0
    # Extract answer from tags first
    pred_best_answer = extract_best_answer(completion)
    pred_worst_answer = extract_worst_answer(completion)
    if pred_best_answer is None or pred_worst_answer is None:
        return 0.0, pred_best_answer, pred_worst_answer
    pred_worst_answer_moves = []
    #extract all the moves from pred_worst_answer
    for move in VALID_ACTIONS:
        if move in pred_worst_answer:
            pred_worst_answer_moves.append(move)

    
    # Extract pos_moves and neg_moves
    pos_moves_str = extract_pos_neg_moves(answer)
    neg_moves_str = extract_neg_moves(answer)
 
    pos_moves = parse_list_str(pos_moves_str) if pos_moves_str else []
    neg_moves = parse_list_str(neg_moves_str) if neg_moves_str else []
    
    # Extract the gold answer
    if '<answer>' in answer:
        gold_match = re.search(r"<answer>(.*?)</answer>", answer, re.DOTALL)
        if gold_match:
            gold_parsed = gold_match.group(1).strip()
        else:
            raise Exception("Failed to parse gold solution: ", answer)
    else:
        gold_parsed = answer.strip()
    pred_best_answer_correct = False
    pred_worst_answer_correct = False
    #handle empty moves
    if len(pred_best_answer.upper()) == 0 and len(pos_moves) == 0:
        pred_best_answer_correct = True
    if pred_worst_answer.upper() == "NONE" and len(neg_moves) == 0:
        pred_worst_answer_correct = True
    #handle correct moves
    
    if pred_worst_answer_moves == neg_moves:
        pred_worst_answer_correct = True
    if pred_best_answer.upper() in pos_moves:
        pred_best_answer_correct = True


    if pred_best_answer_correct and pred_worst_answer_correct:
        reward = 1.0
    else:
        reward = 0.0
        
    return reward, pred_best_answer, pred_worst_answer

def format_reward_func(completion, **kwargs):
    """
    Checks if the completion has the correct format with <think>, <best_answer>, and <worst_answer> tags,
    handling newline characters within the tags.
    """
    pattern = (
        r"^(?=(?:.*<think>){1})(?=(?:.*<\/think>){1})"
        r"(?=(?:.*<best_answer>){1})(?=(?:.*<\/best_answer>){1})"
        r"(?=(?:.*<worst_answer>){1})(?=(?:.*<\/worst_answer>){1})"
        r"(?!.*<think>.*<think>)"
        r"(?!.*<\/think>.*<\/think>)"
        r"(?!.*<best_answer>.*<best_answer>)"
        r"(?!.*<\/best_answer>.*<\/best_answer>)"
        r"(?!.*<worst_answer>.*<worst_answer>)"
        r"(?!.*<\/worst_answer>.*<\/worst_answer>)"
        r".*<think>(.*?)</think>\s*<best_answer>(.*?)</best_answer>\s*<worst_answer>(.*?)</worst_answer>.*$"
    )
    matches = re.search(pattern, completion, re.DOTALL)

    if matches:
        return 0.1
    return 0.0

def reward_func(queries, prompts, labels):
    # queries is prompts + responses
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    rewards = []
    accuracy_rewards = []
    format_rewards = []
    with open(LOG_PATH, "a") as f:
        f.write(f"----------------------------- {current_time} -----------------------------\n")
        for query, prompt, answer in zip(queries, prompts, labels):
            try:
                response = get_response_from_query(query)
                if response == "":
                    f.write("Error: " + query + "\n")
                    rewards.append(0.0)
                    accuracy_rewards.append(0.0)
                    format_rewards.append(0.0)

                else:
                    query1 = get_query_from_query(query)

                    accuracy_reward, pred_best_answer, pred_worst_answer = accuracy_reward_func(response, answer)
                    format_reward = format_reward_func(response)

                    rewards.append(accuracy_reward + format_reward)
                    accuracy_rewards.append(accuracy_reward)
                    format_rewards.append(format_reward)
                    f.write(f"===============================================================\n")
                    f.write("Query: " + query1 + "\n")
                    f.write("Response: " + response + "\n")
                    f.write("Answer: " + answer + "\n")
                    f.write(f"Accuracy Reward: {accuracy_reward}\tFormat Reward: {format_reward}\n\n\n\n")
                    f.write(f"===============================================================\n")
            except:
                f.write("Error: " + query + "\n")
                rewards.append(0.0)
                accuracy_rewards.append(0.0)
                format_rewards.append(0.0)

    return {
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "accuracy_rewards": torch.tensor(accuracy_rewards, dtype=torch.float32),
        "format_rewards": torch.tensor(format_rewards, dtype=torch.float32),
    }
