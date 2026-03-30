import os
import json
import re
import random

DATASET = "katchers"
BASE_DIR = f"./data/{DATASET}"
SAVE_DIR = os.path.join(BASE_DIR, "diverse_profile")
os.makedirs(SAVE_DIR, exist_ok=True)

DIVERSE_NUM = 1   # ⚠️ 처음엔 1로! (시간 절약)
MAX_ITEMS = 4


def clean_text(text: str):
    text = text.replace("nan", "")
    text = re.sub(r"\[.*?\]", "", text)   # [더블딜] 제거
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_items(profile: str):
    items = profile.split(" ; ")
    items = [clean_text(x.strip()) for x in items if x.strip()]
    return items


def shorten_item(item: str):
    item = item.split("|")[0].strip()
    return item


def dedup(items):
    seen = set()
    result = []
    for i in items:
        if i not in seen:
            seen.add(i)
            result.append(i)
    return result


def make_diverse(items):
    items = [shorten_item(x) for x in items]
    items = dedup(items)

    if len(items) == 0:
        return [""]

    if len(items) == 1:
        return [items[0]]

    # 앞 / 뒤 / 섞기 방식
    front = items[:MAX_ITEMS]
    back = items[-MAX_ITEMS:]
    mixed = items[::2] + items[1::2]

    candidates = [
        " ; ".join(front),
        " ; ".join(back),
        " ; ".join(mixed[:MAX_ITEMS]),
    ]

    return candidates[:DIVERSE_NUM]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_profiles(data, id_key, prefix):
    files = [[] for _ in range(DIVERSE_NUM)]

    for id_str, profile in data.items():
        idx = int(id_str)
        items = split_items(profile)
        diverse_list = make_diverse(items)

        for i in range(DIVERSE_NUM):
            files[i].append({
                id_key: idx,
                "profile": diverse_list[i]
            })

    for i in range(DIVERSE_NUM):
        path = os.path.join(SAVE_DIR, f"{prefix}_{i}.json")
        with open(path, "w", encoding="utf-8") as f:
            for row in files[i]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"saved: {path}")


def main():
    user = load_json(os.path.join(BASE_DIR, "user_profile.json"))
    item = load_json(os.path.join(BASE_DIR, "item_profile.json"))

    save_profiles(user, "user_id", "diverse_user_profile")
    save_profiles(item, "item_id", "diverse_item_profile")


if __name__ == "__main__":
    main()