import json
from datasets import Dataset


def load_dyck_json(path: str) -> Dataset:
    with open(path, "r") as f:
        raw = json.load(f)

    prompts = []
    answers = []
    task_ids = []
    difficulties = []

    for item in raw:
        q = item["question"]
        a = item["answer"]
        prompts.append(q)
        answers.append(a)
        task_ids.append(item.get("task_id"))
        diff = item.get("game_data", {}).get("difficulty")
        difficulties.append(diff)

    ds = Dataset.from_dict(
        {
            "prompt": prompts,
            "answer": answers,
            "task_id": task_ids,
            "difficulty": difficulties,
        }
    )
    return ds
