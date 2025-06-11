import os
import re
from datetime import datetime
import sys
import torch
import string
import wandb
import torch.distributed as dist
import ast
import json
LOG_PATH = "reward_test.jsonl"
DEBUG_PATH = os.environ.get("DEBUG_PATH", "debug.log")
# Define valid actions
VALID_ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
WANDB_PROJECT_NAME = "vis"
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

def extract_axis_with_tags(text, input_type):
    if input_type == "completion":
        match = re.search(r'<rotation_axis>(.*?)</rotation_axis>', text, re.DOTALL)
    else:
        match = re.search(r'<axis>(.*?)</axis>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def extract_angle_with_tags(text, input_type):
    if input_type == "completion":
        match = re.search(r'<rotation_angle>(.*?)</rotation_angle>', text, re.DOTALL)
    elif input_type == "answer":
        match = re.search(r'<angle>(.*?)</angle>', text, re.DOTALL)
    else:
        raise ValueError(f"Invalid input type: {input_type}")
    if match:
        return match.group(1).strip()
    with open(DEBUG_PATH, "a") as f:
        f.write(f"no match for input type {input_type} with text {text}")
        
    return None

def extract_numeric_part(angle_str):
    """
    Extract only the numeric part from an angle string.
    For example: "45°" -> "45", "90 degrees" -> "90", "45.5°" -> "45.5", "-30°" -> "-30", etc.
    """
    if angle_str is None:
        return ""
    # Match valid number formats (including negative numbers)
    match = re.search(r'-?\d+(\.\d+)?', angle_str)
    if match:
        return match.group(0).strip()  # Return the full match
    return ""  # Return empty string if no numeric part found
def extract_image_path(answer):
    match_1 = re.search(r'<image_ini_path>(.*?)</image_ini_path>', answer, re.DOTALL)
    match_2 = re.search(r'<image_rot_path>(.*?)</image_rot_path>', answer, re.DOTALL)
    if match_1 and match_2:
        return match_1.group(1).strip(), match_2.group(1).strip()
    elif match_1 == None and match_2 == None:
        return None, None
    else:
        raise ValueError(f"Invalid answer: {answer}")
    
    

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

def accuracy_reward_func_snake(completion, answer):
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
    
def accuracy_reward_func_3d(completion, answer):
    # Extract answer from tags first
    pred_angle = extract_answer_with_tags(completion)
    
    gt_angle = extract_answer_with_tags(answer)
    
    if pred_angle is None:
        return 0.0, ""
    
    # numeric_pred_angle = extract_numeric_part(pred_angle)
    pred_angle = pred_angle.replace("\n", "")
    if pred_angle == gt_angle:
        return 1.0, pred_angle
    else:
        return 0.0, pred_angle
    
def format_reward_func_snake(completion, **kwargs):
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
    
def format_reward_func_3d(completion, **kwargs):
    # Check if the completion has the correct format with <thinking>, <rotation_axis>, and <rotation_angle> tags
    pattern = (
        r"^(?=(?:.*<think>){1})(?=(?:.*<\/think>){1})"
        r"(?=(?:.*<answer>){1})(?=(?:.*<\/answer>){1})"
        r"(?!.*<think>.*<think>)"
        r"(?!.*<\/think>.*<\/think>)"
        r"(?!.*<answer>.*<answer>)"
        r"(?!.*<\/answer>.*<\/answer>)"
        r".*<think>(.+?)</think>\s*<answer>(.+?)</answer>.*$"
    )
    matches = re.search(pattern, completion, re.DOTALL)
    
    # Check if the format is correct
    if matches:
        return 0.1
    return 0.0


def reward_func(queries, prompts, labels):
    # queries is prompts + responses
    rewards = []
    accuracy_rewards = []
    format_rewards = []
    
    with open(LOG_PATH, "a") as f:
        for query, prompt, answer in zip(queries, prompts, labels):
            try:
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                response = get_response_from_query(query)
                
                # 创建记录条目
                log_entry = {
                    "timestamp": current_time,
                }
                
                if response == "":
                    log_entry["query"] = query
                    log_entry["error"] = "Empty response"
                    rewards.append(0.0)
                    accuracy_rewards.append(0.0)
                    format_rewards.append(0.0)
                else:
                    query1 = get_query_from_query(query)
                    log_entry["query"] = query1
                    log_entry["response"] = response
                    log_entry["answer"] = answer
                    
                    if "<image_ini_path>" in answer:
                        accuracy_reward, answer_parsed = accuracy_reward_func_3d(response, answer)
                        format_reward = format_reward_func_3d(response)
                        
                        log_entry["accuracy_reward"] = accuracy_reward
                        log_entry["format_reward"] = format_reward
                        log_entry["type"] = "3d"
                        
                        rewards.append(accuracy_reward + format_reward)
                        accuracy_rewards.append(accuracy_reward)
                        format_rewards.append(format_reward)
                    else:
                        accuracy_reward, pred_pos_moves, pred_neg_moves = accuracy_reward_func_snake(response, answer)
                        format_reward = format_reward_func_snake(response)
                        
                        log_entry["accuracy_reward"] = accuracy_reward
                        log_entry["format_reward"] = format_reward
                        log_entry["type"] = "snake"
                        log_entry["pred_pos_moves"] = pred_pos_moves
                        log_entry["pred_neg_moves"] = pred_neg_moves
                        
                        rewards.append(accuracy_reward + format_reward)
                        accuracy_rewards.append(accuracy_reward)
                        format_rewards.append(format_reward)
            except Exception as e:
                log_entry = {
                    "timestamp": current_time,
                    "query": query,
                    "error": str(e)
                }
                rewards.append(0.0)
                accuracy_rewards.append(0.0)
                format_rewards.append(0.0)
            
            # 将JSON对象写入单行
            f.write(json.dumps(log_entry) + "\n")
    
    return {
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "accuracy_rewards": torch.tensor(accuracy_rewards, dtype=torch.float32),
        "format_rewards": torch.tensor(format_rewards, dtype=torch.float32),
    }