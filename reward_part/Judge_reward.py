from transformers import pipeline

# 使用 facebook/bart-large-mnli 模型初始化 judge model 的 zero-shot-classification pipeline
judge_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

#judge_classifier = pipeline(
#    "zero-shot-classification",
#    model="cross-encoder/nli-distilroberta-base"
#)
import numpy as np
def add_gaussian_perturbation(x, sigma=0.1):
    perturbation = np.random.normal(0, sigma)
    x_perturbed = x + perturbation
    return np.clip(x_perturbed, 0.0, 1.0)


def compute_judge_score(question: str, candidate: str, reference: str) -> float:
    """
    使用 judge model 判断候选答案是否与 ground truth 匹配，并返回归一化的得分（范围 0 到 1）。

    构造拼接的 prompt，将问题、候选答案和参考答案整合后提问：
    "Does the candidate answer match the ground truth? Answer yes or no."
    
    Args:
        question (str): 问题文本
        candidate (str): 候选答案
        reference (str): 参考答案（ground truth）
    
    Returns:
        float: 归一化得分，取 "yes" 的概率，值域在 0 到 1 之间。
    """
    prompt = (
        f"Question: {question}\n"
        f"Candidate Answer: {candidate}\n"
        f"Ground Truth: {reference}\n"
        "Does the candidate answer match the ground truth? Answer yes or no."
    )
    candidate_labels = ["yes", "no"]
    
    # 使用零样本分类模型进行判断
    result = judge_classifier(prompt, candidate_labels=candidate_labels)
    
    # 提取模型返回中 "yes" 对应的概率，该概率已经归一化到 0 到 1
    yes_score = 0.0
    for label, score in zip(result["labels"], result["scores"]):
        if label.lower() == "yes":
            yes_score = score
            break
            
    return yes_score

def evaluate_responses1(question: str, reference: str, candidate: str) -> float:
    """
    对单个候选回答进行评估，利用 judge model 计算候选答案与参考答案匹配的得分。

    Args:
        question (str): 问题文本
        reference (str): 参考答案（ground truth）
        candidate (str): 候选回答
    
    Returns:
        float: Judge 得分（正值表示候选答案匹配程度较好）
    """
    #print("Evaluating candidate answer...")
    #print("Question:", question)
    #print("Candidate Answer:", candidate)
    #print("Ground Truth:", reference)
    
    score = compute_judge_score(question, candidate, reference)
    
    print("Judge Score:", score)
    return score

