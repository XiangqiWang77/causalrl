import asyncio
import datetime
import json
import math
import os
import random
import re
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging
from rouge_score import rouge_scorer

# import logging
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from logging.handlers import TimedRotatingFileHandler

from pydantic import BaseModel
import re
from typing import List
from math_verify import LatexExtractionConfig, StringExtractionConfig, parse, verify
import requests


def extract_answer(text):
    """ """
    pattern = r"<finalanswer>(.*?)</finalanswer>"
    matches = re.findall(pattern, text, re.DOTALL)
    if len(matches) == 1:
        return matches[0]
    else:
        return None


def extract_answer_deepseek(text):
    """ """
    pattern = r"</think>\n(.*?)<｜end▁of▁sentence｜>"
    matches = re.findall(pattern, text, re.DOTALL)
    if len(matches) == 1:
        return matches[0]
    else:
        return None


def extract_user_query_deepseek(text):
    """ """
    pattern = r"<｜User｜>(.*?)<｜Assistant｜><think>\n"
    matches = re.findall(pattern, text, re.DOTALL)
    if len(matches) == 1:
        return matches[0]
    else:
        return None


def extract_boxed(text):
    """使用计数方法提取\boxed{}中的内容，处理任意嵌套的大括号"""
    if not isinstance(text, str):
        return None

    boxed_start = text.find(r"\boxed{")
    if boxed_start == -1:
        return None

    # 跳过'\boxed{'
    start_pos = boxed_start + 7
    brace_count = 1
    end_pos = start_pos

    # 通过计数大括号来找到匹配的结束大括号
    while end_pos < len(text) and brace_count > 0:
        if text[end_pos] == "{":
            brace_count += 1
        elif text[end_pos] == "}":
            brace_count -= 1
        end_pos += 1

    if brace_count == 0:
        # 减去1是因为我们要排除最后的大括号
        return text[start_pos : end_pos - 1]

    return None


def sub_answer_by_math(x: List[str] | str | None):
    if x is None:
        return []
    if isinstance(x, str):
        x = [x]

    res = [] + x

    select_config = [
        StringExtractionConfig(),
        LatexExtractionConfig(),
        # LatexExtractionConfig(),
    ]

    for temp_x in x:
        res.extend(parse(temp_x, extraction_mode="first_match"))
        for temp_config in select_config:
            res.extend(
                parse(
                    temp_x,
                    extraction_mode="first_match",
                    extraction_config=[temp_config],
                )
            )

    res = list(set(res))
    return res


def get_model_gen_result(x: str):
    # 用于提取模型输出的最终结果
    if not isinstance(x, str):
        return []
    else:
        answer_part = extract_answer_deepseek(x)
        if answer_part is None:
            return []
        else:
            box_part = extract_boxed(answer_part)
            box_part = [] if box_part is None else box_part

            math_value = sub_answer_by_math(box_part)
            # math_value.extend()

            math_value = list(set(math_value))
            return math_value


def get_ground_truth(x: str):
    # 处理ground truth的结果
    res = sub_answer_by_math(x)
    res = list(set(res))
    return res


def calc_accuracy4math(ground_truth: str, response: str) -> float:
    """计算数学公式的准确率"""
    # 提取模型输出的最终结果
    model_gen_result = get_model_gen_result(response)
    # 提取标准答案的最终结果
    ground_truth_result = get_ground_truth(ground_truth)

    # 计算准确率
    accuracy = verify(model_gen_result, ground_truth_result)
    accuracy = accuracy * 1.0
    return accuracy


def get_cloud_score_api(query: str, response: str):
    try:
        api_url = "http://10.xxx.0.xxx:7009/pooling"

        # Input like Chat API
        prompt = {
            "model": "r1-reward",
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are a helpful assistant."}
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": query}],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": response}],
                },
            ],
        }
        headers = {"Content-Type": "application/json", "Authorization": "Bearer EMPTY"}
        response = requests.post(api_url, headers=headers, json=prompt)

        # score = {"score": response.json()["data"][0]["data"][0]}
        score = response.json()["data"][0]["data"][0]
        score = float(score)

        return score
    except Exception as e:
        return 0.0


def get_reward_score_api_cloud(user_prompt: str, response_str):
    response = requests.post(
        "http://10.xxx.240.xxx:5008/reward/api",
        json={"user_prompt": user_prompt, "response": response_str},
    )
    return response.json()


def calc_cloud_score(query: str, response: str) -> float:
    """使用自定义的reward接口进行训练"""
    model_response_answer = extract_answer_deepseek(response)
    query = extract_user_query_deepseek(query)

    if query is None:
        return -99.0, None
    if model_response_answer is None:
        return -99.0, None
    else:
        score = get_reward_score_api_cloud(query, model_response_answer)
        return float(score.get("reward_score")), str(score.get("critique"))

logger = logging.getLogger(__name__)


file_handler = TimedRotatingFileHandler(
    "data/reward_log/reward.log",
    when="midnight",
    interval=1,
    backupCount=5,
)
file_handler.setLevel(logging.INFO)  # 设置文件处理器的日志级别
logger.addHandler(file_handler)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class BaseRequest(BaseModel):
    data_source: Any
    solution_str: str
    ground_truth: Union[dict, list, str]
    extra_info: Union[dict, list, str]


def format_reward(answer: str | List[str], **kwargs) -> List[float]:
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <finalanswer> and </finalanswer> tags."""

    # step 1
    pattern = r"^<think>.*?</think>\s*<finalanswer>.*?</finalanswer>$"

    def drop_end_tag(x: str):
        end_tag_list = ["<|im_end|>", "<|endoftext|>"]

        for temp_end_tag in end_tag_list:
            if x.endswith(temp_end_tag):
                value = x[: -len(temp_end_tag)]
                return value

        value = x
        return value

    if isinstance(answer, list):
        completion_contents = [drop_end_tag(x) for x in answer]
    else:
        completion_contents = [drop_end_tag(answer)]

    matches = [
        re.match(pattern, content, re.DOTALL | re.MULTILINE)
        for content in completion_contents
    ]
    # step1_score = [1.0 if match else 0.0 for match in matches]

    # step 2
    def format_func_patch1(x: str):
        # Check if the string starts with <think> and ends with </finalanswer>
        if not (x[:100].find("<think>") != -1 and x[-100:].find("</finalanswer>") != -1):
            # if not (x[:100].startswith("<think>") and x[-100:].endswith("</finalanswer>")):
            return 0.0

        # Check if each tag appears exactly once
        tags = ["<think>", "</think>", "<finalanswer>", "</finalanswer>"]
        for tag in tags:
            if x.count(tag) != 1:
                return 0.0

        # Check the order of tags
        pos_think_start = x.find("<think>")
        pos_think_end = x.find("</think>")
        pos_answer_start = x.find("<finalanswer>")
        pos_answer_end = x.find("</finalanswer>")

        # Verify correct order
        if not (pos_think_start < pos_think_end < pos_answer_start < pos_answer_end):
            return 0.0

        return 1.0

    final_res = []
    for content_ in completion_contents:
        score = format_func_patch1(content_)
        final_res.append(score)
    # for content_, match in zip(completion_contents, matches):
    #     if match:
    #         score = format_func_patch1(content_)
    #         final_res.append(score)
    #     else:
    #         final_res.append(0.0)

    return final_res


def format_reward_deepseek_wrap(answer: str | List[str], **kwargs) -> List[float]:
    "优化适配deepseek 模型的format reward"
    if isinstance(answer, str):
        return format_reward_deepseek(answer)
    elif isinstance(answer, list):
        return [format_reward_deepseek(x) for x in answer]


def accuracy_reward(pred_answer: List[str], ground_truth: List[str], **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""

    rewards = []
    for pred, truth in zip(pred_answer, ground_truth):
        temp_value = calc_accuracy4math(truth, pred)
        rewards.append(temp_value)

    return rewards


@app.post("/get_reward")
async def get_reward(request: Request):
    json_data = await request.json()

    # print(json_data)

    groud_truth: str = json_data.get("ground_truth", "")
    pred_answer = json_data.get("response_str", "")

    if isinstance(pred_answer, list):
        pred_answer = pred_answer[0]

    # 模拟打分
    score = {
        "format": format_reward([pred_answer])[0],
        "accuracy": accuracy_reward([pred_answer], [groud_truth])[0],
        # "relevance": random.randint(0, 1),
    }
    score["score"] = sum(
        score.values()
    )  # 注意，最后返回的是一个字典。然后这个字典里面一定要有score这个key。值为0 或者1

    cur_date = datetime.datetime.now()

    temp_data = {
        "cur_date": cur_date.strftime("%Y-%m-%d %X"),
        "input_data": json_data,
        "score": score,
    }
    logger.info(json.dumps(temp_data, ensure_ascii=False))

    return score


@app.post("/get_reward2")
async def get_reward2(request: Request):
    """使用自定义的reward模型，提供反馈接口"""
    json_data = await request.json()

    def wrap_process_data():
        from signal_utils import extract_XY
        
        import torch
        import torch.nn.functional as F
        from transformers import AutoTokenizer, AutoModelForCausalLM

        # 1) get Z and raw model output
        Z = json_data.get("prompt_str", "")
        resp = json_data.get("response_str", "")
        if isinstance(resp, list):
            resp = resp[0]

        # 2) split into X (CoT) and Y (final answer)
        X, Y = extract_XY(resp)

        print("Z:", Z)
        print("X:", X)
        print("Y:", Y)

        
        #print(a)
        # 3) pick your Minkowski norm order
        p = float(json_data.get("p_norm", 2.0))

        
        s_ZX, s_ZY, s_XY=calc.compute(Z, X, Y)
        
        # 4) compute the four causal‐Jacobian signals

        # 5) Minkowski‐p combination of them
        total = (abs(s_ZX)**p + abs(s_ZY)**p + abs(s_XY)**p)**(1.0/p)

        result = {
            "S(Z→X)": round(s_ZX, 6),
            "S(Z→Y)": round(s_ZY, 6),
            "S(X→Y)": round(s_XY, 6),
            "score": round(total, 6),
        }

        logger.info(json.dumps({"input_data": json_data, "reward": result}, ensure_ascii=False))
        return result

    # Use asyncio to offload potentially CPU-bound task to a thread
    result = await asyncio.to_thread(wrap_process_data)
    return result

from modi_signal import RMSNormalizedSignalCalculator
model_dir = "/users/xwang76/hf_models/qwen3-4b"
calc =RMSNormalizedSignalCalculator(model_dir, device="cuda")
if __name__ == "__main__":
    import uvicorn

    
    uvicorn.run(app, host="0.0.0.0", port=6009)
