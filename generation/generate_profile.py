import os
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# ===== 설정 =====
MODEL_NAME = "gpt-4.1-mini"
MAX_WORKERS = 8
SAVE_EVERY = 100
TEMPERATURE = 0.2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "katchers")
SAVE_DIR = os.path.join(DATA_DIR, "llm_profiles")
INSTRUCTION_DIR = os.path.join(BASE_DIR, "instruction")

os.makedirs(SAVE_DIR, exist_ok=True)

client = OpenAI()
save_lock = threading.Lock()

ITEM_OUTPUT_PATH = os.path.join(SAVE_DIR, "item_profile_llm.json")
USER_OUTPUT_PATH = os.path.join(SAVE_DIR, "user_profile_llm.json")
ITEM_FAIL_PATH = os.path.join(SAVE_DIR, "item_profile_llm_failed.json")
USER_FAIL_PATH = os.path.join(SAVE_DIR, "user_profile_llm_failed.json")
ITEM_CACHE_PATH = os.path.join(SAVE_DIR, "item_prompt_cache.json")
USER_CACHE_PATH = os.path.join(SAVE_DIR, "user_prompt_cache.json")


def load_json(path, default=None):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {} if default is None else default


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_instruction(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def is_empty_text(text):
    return text is None or not str(text).strip()


def make_cache_key(system_prompt, prompt):
    return f"{system_prompt}\n\n###INPUT###\n{prompt}"


def save_item_state(item_profiles_llm, item_failed, item_prompt_cache):
    save_json(ITEM_OUTPUT_PATH, item_profiles_llm)
    save_json(ITEM_FAIL_PATH, item_failed)
    save_json(ITEM_CACHE_PATH, item_prompt_cache)


def save_user_state(user_profiles_llm, user_failed, user_prompt_cache):
    save_json(USER_OUTPUT_PATH, user_profiles_llm)
    save_json(USER_FAIL_PATH, user_failed)
    save_json(USER_CACHE_PATH, user_prompt_cache)


item_system_prompt = load_instruction(
    os.path.join(INSTRUCTION_DIR, "item_system_prompt.txt")
)
user_system_prompt = load_instruction(
    os.path.join(INSTRUCTION_DIR, "user_system_prompt.txt")
)

with open(os.path.join(DATA_DIR, "item_profile.json"), "r", encoding="utf-8") as f:
    item_profiles = json.load(f)

with open(os.path.join(DATA_DIR, "user_profile.json"), "r", encoding="utf-8") as f:
    user_profiles = json.load(f)

item_profiles_llm = load_json(ITEM_OUTPUT_PATH, {})
user_profiles_llm = load_json(USER_OUTPUT_PATH, {})
item_failed = load_json(ITEM_FAIL_PATH, {})
user_failed = load_json(USER_FAIL_PATH, {})
item_prompt_cache = load_json(ITEM_CACHE_PATH, {})
user_prompt_cache = load_json(USER_CACHE_PATH, {})

print(f"기존 item 결과 수: {len(item_profiles_llm)}")
print(f"기존 user 결과 수: {len(user_profiles_llm)}")
print(f"item 캐시 수: {len(item_prompt_cache)}")
print(f"user 캐시 수: {len(user_prompt_cache)}")


def get_response(system_prompt, prompt):
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=TEMPERATURE,
    )
    content = completion.choices[0].message.content
    return "" if content is None else content.strip()


def prepare_item_jobs():
    jobs = []
    skipped_empty = 0
    cache_hit = 0

    for item_id, text in item_profiles.items():
        item_id = str(item_id)

        if item_id in item_profiles_llm and not is_empty_text(item_profiles_llm[item_id]):
            continue

        if is_empty_text(text):
            item_profiles_llm[item_id] = ""
            item_failed[item_id] = {"reason": "empty_input"}
            skipped_empty += 1
            continue

        cache_key = make_cache_key(item_system_prompt, text)
        if cache_key in item_prompt_cache and not is_empty_text(item_prompt_cache[cache_key]):
            item_profiles_llm[item_id] = item_prompt_cache[cache_key]
            item_failed.pop(item_id, None)
            cache_hit += 1
            continue

        jobs.append((item_id, text, cache_key))

    return jobs, cache_hit, skipped_empty


def prepare_user_jobs():
    jobs = []
    skipped_empty = 0
    skipped_fallback = 0
    cache_hit = 0

    for user_id, text in user_profiles.items():
        user_id = str(user_id)

        if user_id in user_profiles_llm and not is_empty_text(user_profiles_llm[user_id]):
            continue

        if text == "no purchase profile":
            user_profiles_llm[user_id] = ""
            user_failed[user_id] = {"reason": "fallback_profile"}
            skipped_fallback += 1
            continue

        if is_empty_text(text):
            user_profiles_llm[user_id] = ""
            user_failed[user_id] = {"reason": "empty_input"}
            skipped_empty += 1
            continue

        cache_key = make_cache_key(user_system_prompt, text)
        if cache_key in user_prompt_cache and not is_empty_text(user_prompt_cache[cache_key]):
            user_profiles_llm[user_id] = user_prompt_cache[cache_key]
            user_failed.pop(user_id, None)
            cache_hit += 1
            continue

        jobs.append((user_id, text, cache_key))

    return jobs, cache_hit, skipped_empty, skipped_fallback


def process_one_item(job):
    item_id, text, cache_key = job
    try:
        response = get_response(item_system_prompt, text)
        if is_empty_text(response):
            return {
                "id": item_id,
                "ok": False,
                "response": "",
                "cache_key": cache_key,
                "error_type": "empty_response",
                "input_preview": text[:300],
            }
        return {
            "id": item_id,
            "ok": True,
            "response": response,
            "cache_key": cache_key,
        }
    except Exception as e:
        return {
            "id": item_id,
            "ok": False,
            "response": "",
            "cache_key": cache_key,
            "error_type": "exception",
            "error": str(e),
            "input_preview": text[:300],
        }


def process_one_user(job):
    user_id, text, cache_key = job
    try:
        response = get_response(user_system_prompt, text)
        if is_empty_text(response):
            return {
                "id": user_id,
                "ok": False,
                "response": "",
                "cache_key": cache_key,
                "error_type": "empty_response",
                "input_preview": text[:300],
            }
        return {
            "id": user_id,
            "ok": True,
            "response": response,
            "cache_key": cache_key,
        }
    except Exception as e:
        return {
            "id": user_id,
            "ok": False,
            "response": "",
            "cache_key": cache_key,
            "error_type": "exception",
            "error": str(e),
            "input_preview": text[:300],
        }


print("=== ITEM PROFILE 생성 ===")
item_jobs, item_cache_hit_count, item_skipped_empty_count = prepare_item_jobs()
print(f"item 실제 API 호출 예정 수: {len(item_jobs)}")
print(f"item 캐시 재사용 수: {item_cache_hit_count}")
print(f"item 빈 입력 스킵 수: {item_skipped_empty_count}")

item_processed_count = 0

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(process_one_item, job) for job in item_jobs]

    for future in as_completed(futures):
        result = future.result()
        item_id = result["id"]

        if result["ok"]:
            item_profiles_llm[item_id] = result["response"]
            item_prompt_cache[result["cache_key"]] = result["response"]
            item_failed.pop(item_id, None)
        else:
            item_profiles_llm[item_id] = ""
            item_failed[item_id] = {
                "reason": result["error_type"],
                "error": result.get("error", ""),
                "input_preview": result.get("input_preview", ""),
            }

        item_processed_count += 1

        if item_processed_count % 10 == 0:
            print(f"[ITEM] 완료={item_processed_count}/{len(item_jobs)}", flush=True)

        if item_processed_count % SAVE_EVERY == 0:
            with save_lock:
                save_item_state(item_profiles_llm, item_failed, item_prompt_cache)
            print(f"[ITEM] 중간 저장 완료 ({item_processed_count}개)", flush=True)

with save_lock:
    save_item_state(item_profiles_llm, item_failed, item_prompt_cache)

print("item 완료")
print(f"item 실제 API 호출 수: {item_processed_count}")
print(f"item 실패 수: {len(item_failed)}")


print("=== USER PROFILE 생성 ===")
user_jobs, user_cache_hit_count, user_skipped_empty_count, user_skipped_fallback_count = prepare_user_jobs()
print(f"user 실제 API 호출 예정 수: {len(user_jobs)}")
print(f"user 캐시 재사용 수: {user_cache_hit_count}")
print(f"user fallback 스킵 수: {user_skipped_fallback_count}")
print(f"user 빈 입력 스킵 수: {user_skipped_empty_count}")

user_processed_count = 0

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(process_one_user, job) for job in user_jobs]

    for future in as_completed(futures):
        result = future.result()
        user_id = result["id"]

        if result["ok"]:
            user_profiles_llm[user_id] = result["response"]
            user_prompt_cache[result["cache_key"]] = result["response"]
            user_failed.pop(user_id, None)
        else:
            user_profiles_llm[user_id] = ""
            user_failed[user_id] = {
                "reason": result["error_type"],
                "error": result.get("error", ""),
                "input_preview": result.get("input_preview", ""),
            }

        user_processed_count += 1

        if user_processed_count % 10 == 0:
            print(f"[USER] 완료={user_processed_count}/{len(user_jobs)}", flush=True)

        if user_processed_count % SAVE_EVERY == 0:
            with save_lock:
                save_user_state(user_profiles_llm, user_failed, user_prompt_cache)
            print(f"[USER] 중간 저장 완료 ({user_processed_count}개)", flush=True)

with save_lock:
    save_user_state(user_profiles_llm, user_failed, user_prompt_cache)

print("user 완료")
print(f"user 실제 API 호출 수: {user_processed_count}")
print(f"user 실패 수: {len(user_failed)}")