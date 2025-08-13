import os
import requests

# DeepInfra 配置
DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")

def deepinfra_chat(messages, model="meta-llama/Meta-Llama-3.1-8B-Instruct", temperature=0.7):
    url = f"{DEEPINFRA_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPINFRA_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()

def evaluate_responses1(question: str, response: str, ground_truth: str) -> float:
    """
    Use an LLM to judge whether `response` matches `ground_truth`.
    Returns 1 if the model answers 'yes', otherwise 0.
    """
    response=response[:512]
    ground_truth=ground_truth[:256]
    prompt = (
        f"Question: {question}\n\n"
        f"Candidate Answer: {response}\n\n"
        f"Ground Truth: {ground_truth}\n\n"
        "Does the candidate answer match the ground truth?  If you are not certain of yes or no, reply ambiguous. Reply with only 'yes' or 'no'or 'ambiguous'. Don't generate anything else! Just yes or no or ambiguous. Please be noted that ambiguous and no should be used sparingly only if you are certain about it."
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": prompt}
    ]
    #print(messages)
    #If the candidate answer is semantically correct, close in meaning, or approximately matches the ground truth, output yes. Accept partial correctness, paraphrasing, minor numerical or formatting differences, and extra non-conflicting details as matches. Output no only for clear, significant deviations or contradictions. If you are not certain of yes or no, reply ambiguous. Reply with only 'yes' or 'no'or 'ambiguous'. Don't generate anything else! Just yes or no or ambiguous."
    model_reply = deepinfra_chat(messages)["choices"][0]["message"]["content"].strip().lower()
    #print(model_reply)
    if model_reply == "yes":
        return 1.0
    elif model_reply == "no":
        return 0.0
    else:
        return 0.5
