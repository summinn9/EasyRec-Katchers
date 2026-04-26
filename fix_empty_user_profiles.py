import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(BASE_DIR, "data", "katchers", "llm_profiles", "user_profile_llm_orderaware.json")

with open(path, "r", encoding="utf-8") as f:
    user_profiles = json.load(f)

fixed = 0

for uid, text in user_profiles.items():
    if text is None or str(text).strip() == "":
        user_profiles[uid] = '{"summarization": "User with insufficient purchase history.", "reasoning": "No training purchase profile was available for this user."}'
        fixed += 1

with open(path, "w", encoding="utf-8") as f:
    json.dump(user_profiles, f, ensure_ascii=False, indent=2)

print("빈 user profile 수정 수:", fixed)