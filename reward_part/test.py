import requests


def call_online_reward_model(**kwargs):
    try:
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
        }

        json_data = {**kwargs}

        response = requests.post(
            "http://0.0.0.0:6010/get_reward2", headers=headers, json=json_data
        )
        res = response.json()

        final_score = res.get("score")
        print(res)
        return float(final_score)
    except Exception as e:
        print(e)
        return 0.0


input_data = {
    "data_source": "DigitalLearningGmbH/MATH-lighteval",
    "solution_str": "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nIf $A$ is an angle such that $\\tan A + \\sec A = 2,$ enter all possible values of $\\cos A,$ separated by commas. Let's think step by step and output the final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\nTo solve for \\(\\cos A\\) given that \\(\\tan A + \\sec A = 2\\), we start by expressing \\(\\tan A\\) and \\(\\sec A\\) in terms of \\(\\sin A\\) and \\(\\cos A\\).\n\nWe know:\n\\[\n\\tan A = \\frac{\\sin A}{\\cos A} \\quad \\text{and} \\quad \\sec A = \\frac{1}{\\cos A}\n\\]\nSubstituting these into the given equation:\n\\[\n\\frac{\\sin A}{\\cos A} + \\frac{1}{\\cos A} = 2\n\\]\nCombining the terms on the left-hand side:\n\\[\n\\frac{\\sin A + 1}{\\cos A} = 2\n\\]\nMultiplying both sides by \\(\\cos A\\):\n\\[\n\\sin A + 1 = 2 \\cos A\n\\]\nRearranging to isolate \\(\\sin A\\):\n\\[\n\\sin A = 2 \\cos A - 1\n\\]\n\nNext, we use the Pythagorean identity \\(\\sin^2 A + \\cos^2 A = 1\\). Substituting \\(\\sin A = 2 \\cos A - 1\\) into this identity:\n\\[\n(2 \\cos A - 1)^2 + \\cos^2 A = 1\n\\]\nExpanding and simplifying the left-hand side:\n\\[\n(4 \\cos^2 A - 4 \\cos A + 1) + \\cos^2 A = 1\n\\]\nCombine like terms:\n\\[\n5 \\cos^2 A - 4 \\cos A + 1 = 1\n\\]\nSubtracting 1 from both sides:\n\\[\n5 \\cos^2 A - 4 \\cos A = 0\n\\]\nFactoring out \\(\\cos A\\):\n\\[\n\\cos A (5 \\cos A - 4) = 0\n\\]\n\nThis gives us two possible solutions:\n\\[\n\\cos A = 0 \\quad \\text{or} \\quad 5 \\cos A - 4 = 0\n\\]\n\nSolving the second equation:\n\\[\n5 \\cos A = 4\n\\]\n\\[\n\\cos A = \\frac{4}{5}\n\\]\n\nNow, we need to check which of these solutions satisfy the original equation \\(\\tan A + \\sec A = 2\\).\n\n1. For \\(\\cos A = 0\\):\n   \\[\n   \\sec A = \\frac{1}{\\cos A} = \\frac{1}{0}\n   \\]\n   \\(\\sec A\\) is undefined, so \\(\\cos A = 0\\) is not a valid solution.\n\n2. For \\(\\cos A = \\frac{4}{5}\\):\n   \\[\n   \\sin A = 2 \\cos A - 1 = 2 \\left(\\frac{4}{5}\\right) - 1 = \\frac{8}{5} - 1 = \\frac{3}{5}\n   \\]\n   \\[\n   \\tan A = \\frac{\\sin A}{\\cos A} = \\frac{\\frac{3}{5}}{\\frac{4}{5}} = \\frac{3}{4}\n   \\]\n   \\[\n   \\sec A = \\frac{1}{\\cos A} = \\frac{5}{4}\n   \\]\n   \\[\n   \\tan A + \\sec A = \\frac{3}{4} + \\frac{5}{4} = 2\n   \\]\n   This solution satisfies the original equation.\n\nTherefore, the only valid solution is \\(\\cos A = \\frac{4}{5}\\).\n\nThe final answer is:\n\\[\n\\boxed{\\frac{4}{5}}\n\\]<|im_end|>",
    "ground_truth": "\\frac{4}{5}",
    "extra_info": {"index": 6820, "split": "train"},
    "prompt_str": "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nIf $A$ is an angle such that $\\tan A + \\sec A = 2,$ enter all possible values of $\\cos A,$ separated by commas. Let's think step by step and output the final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n",
    "response_str": [
        "We can conclude that blablablabla <finalanswer>\\boxed{\\frac{4}{5}}</finalanswer>"
    ],  # ["To solve for \\(\\cos A\\) given that \\(\\tan A + \\sec A = 2\\), we start by expressing \\(\\tan A\\) and \\(\\sec A\\) in terms of \\(\\sin A\\) and \\(\\cos A\\).\n\nWe know:\n\\[\n\\tan A = \\frac{\\sin A}{\\cos A} \\quad \\text{and} \\quad \\sec A = \\frac{1}{\\cos A}\n\\]\nSubstituting these into the given equation:\n\\[\n\\frac{\\sin A}{\\cos A} + \\frac{1}{\\cos A} = 2\n\\]\nCombining the terms on the left-hand side:\n\\[\n\\frac{\\sin A + 1}{\\cos A} = 2\n\\]\nMultiplying both sides by \\(\\cos A\\):\n\\[\n\\sin A + 1 = 2 \\cos A\n\\]\nRearranging to isolate \\(\\sin A\\):\n\\[\n\\sin A = 2 \\cos A - 1\n\\]\n\nNext, we use the Pythagorean identity \\(\\sin^2 A + \\cos^2 A = 1\\). Substituting \\(\\sin A = 2 \\cos A - 1\\) into this identity:\n\\[\n(2 \\cos A - 1)^2 + \\cos^2 A = 1\n\\]\nExpanding and simplifying the left-hand side:\n\\[\n(4 \\cos^2 A - 4 \\cos A + 1) + \\cos^2 A = 1\n\\]\nCombine like terms:\n\\[\n5 \\cos^2 A - 4 \\cos A + 1 = 1\n\\]\nSubtracting 1 from both sides:\n\\[\n5 \\cos^2 A - 4 \\cos A = 0\n\\]\nFactoring out \\(\\cos A\\):\n\\[\n\\cos A (5 \\cos A - 4) = 0\n\\]\n\nThis gives us two possible solutions:\n\\[\n\\cos A = 0 \\quad \\text{or} \\quad 5 \\cos A - 4 = 0\n\\]\n\nSolving the second equation:\n\\[\n5 \\cos A = 4\n\\]\n\\[\n\\cos A = \\frac{4}{5}\n\\]\n\nNow, we need to check which of these solutions satisfy the original equation \\(\\tan A + \\sec A = 2\\).\n\n1. For \\(\\cos A = 0\\):\n   \\[\n   \\sec A = \\frac{1}{\\cos A} = \\frac{1}{0}\n   \\]\n   \\(\\sec A\\) is undefined, so \\(\\cos A = 0\\) is not a valid solution.\n\n2. For \\(\\cos A = \\frac{4}{5}\\):\n   \\[\n   \\sin A = 2 \\cos A - 1 = 2 \\left(\\frac{4}{5}\\right) - 1 = \\frac{8}{5} - 1 = \\frac{3}{5}\n   \\]\n   \\[\n   \\tan A = \\frac{\\sin A}{\\cos A} = \\frac{\\frac{3}{5}}{\\frac{4}{5}} = \\frac{3}{4}\n   \\]\n   \\[\n   \\sec A = \\frac{1}{\\cos A} = \\frac{5}{4}\n   \\]\n   \\[\n   \\tan A + \\sec A = \\frac{3}{4} + \\frac{5}{4} = 2\n   \\]\n   This solution satisfies the original equation.\n\nTherefore, the only valid solution is \\(\\cos A = \\frac{4}{5}\\).\n\nThe final answer is:\n\\[\n\\boxed{\\frac{4}{5}}\n\\]<|im_end|>"],
}



call_online_reward_model(**input_data)
