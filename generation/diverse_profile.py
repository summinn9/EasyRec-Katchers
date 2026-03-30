import os
import json
import time
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

DATASET = "katchers"
BASE_DIR = f"./data/{DATASET}"
SAVE_DIR = os.path.join(BASE_DIR, "diverse_profile")
os.makedirs(SAVE_DIR, exist_ok=True)

def get_gpt_response_w_system(instruction, prompt):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt},
        ]
    )
    return completion.choices[0].message.content.strip()

# 원본 profile 읽기: 현재 네 파일 형식은 json.load() 방식
with open(os.path.join(BASE_DIR, "user_profile.json"), "r", encoding="utf-8") as f:
    user_profile = json.load(f)

with open(os.path.join(BASE_DIR, "item_profile.json"), "r", encoding="utf-8") as f:
    item_profile = json.load(f)

# system prompt 읽기
with open("./generation/instruction/user_system_prompt_diverse.txt", "r", encoding="utf-8") as f:
    user_system_prompt = f.read()

with open("./generation/instruction/item_system_prompt_diverse.txt", "r", encoding="utf-8") as f:
    item_system_prompt = f.read()

DIVERSE_NUM = 3

for diverse_no in range(DIVERSE_NUM):
    user_save_path = os.path.join(SAVE_DIR, f"diverse_user_profile_{diverse_no}.json")
    item_save_path = os.path.join(SAVE_DIR, f"diverse_item_profile_{diverse_no}.json")

    with open(user_save_path, "w", encoding="utf-8") as uf:
        for user_id_str, profile in user_profile.items():
            user_id = int(user_id_str)
            response = get_gpt_response_w_system(user_system_prompt, profile)
            record = {"user_id": user_id, "profile": response}
            uf.write(json.dumps(record, ensure_ascii=False) + "\n")
            time.sleep(0.5)

    with open(item_save_path, "w", encoding="utf-8") as itf:
        for item_id_str, profile in item_profile.items():
            item_id = int(item_id_str)
            response = get_gpt_response_w_system(item_system_prompt, profile)
            record = {"item_id": item_id, "profile": response}
            itf.write(json.dumps(record, ensure_ascii=False) + "\n")
            time.sleep(0.5)

print("Diverse profiles saved successfully.")