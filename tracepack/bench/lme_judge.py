"""LongMemEval's official answer-checking prompts, copied verbatim from `src/evaluation/evaluate_qa.py` of
https://github.com/xiaowu0162/LongMemEval (MIT License, Copyright (c) 2024 Di Wu et al.).

`judge()` sends one prompt to the judge model and reads the first standalone "yes" or "no" in its reply
(up to three tries when it says neither).
"""
from __future__ import annotations

import re


def get_anscheck_prompt(task, question, answer, response, abstention=False):
    if not abstention:
        if task in ['single-session-user', 'single-session-assistant', 'multi-session']:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'temporal-reasoning':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'knowledge-update':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'single-session-preference':
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        else:
            raise NotImplementedError
    else:
        template = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
        prompt = template.format(question, answer, response)
    return prompt


_YES_NO = re.compile(r"\b(yes|no)\b")


def judge(endpoint, item: dict, response: str) -> dict:
    """{"ok": bool, "raw": ...} | {"empty": True, ...} | {"err": ...}. The judge sees only the question, the
    reference answer and the response -- never which arm produced it."""
    p = get_anscheck_prompt(item["question_type"], item["question"], item["answer"], response or "",
                            abstention="_abs" in item["question_id"])
    txt = ""
    for k in range(3):
        try:
            txt, _tcs, u = endpoint.chat([{"role": "user", "content": p}], temperature=0.0, max_tokens=256)
        except Exception as e:                    # noqa: BLE001 - recorded, the unit stays ungraded
            return {"err": repr(e)[:200]}
        t = (txt or "").strip().lower()
        m = _YES_NO.search(t)
        if m:
            return {"ok": m.group(1) == "yes", "raw": t[:80], "usage": u, "attempts": k + 1}
    return {"empty": True, "raw": (txt or "")[:80]}
