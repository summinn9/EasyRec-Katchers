import os
import json
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

# =========================
# 설정
# =========================
MODEL_NAME = "gpt-4.1-mini"

DATASET = "katchers"
BASE_DIR = Path(f"./data/{DATASET}")
SAVE_DIR = BASE_DIR / "diverse_profile"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

DIVERSE_NUM = 1
MAX_RETRY = 5
MAX_WORKERS = 4
REQUEST_TIMEOUT = 60

client = OpenAI()

# 파일 쓰기 락
write_lock = threading.Lock()
print_lock = threading.Lock()

# =========================
# 유틸 함수
# =========================
def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_text(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def read_existing_ids(path: Path, id_key: str):
    done_ids = set()
    if not path.exists():
        return done_ids

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                done_ids.add(int(obj[id_key]))
            except Exception:
                continue
    return done_ids

def extract_response_text(text: str) -> str:
    """
    'REVISED PROFILE: ' 접두어가 있으면 제거
    """
    prefix = "REVISED PROFILE:"
    if text.startswith(prefix):
        return text[len(prefix):].strip()
    return text.strip()

def generate_diverse_profile(system_prompt: str, original_profile: str) -> str:
    last_error = None

    for attempt in range(1, MAX_RETRY + 1):
        try:
            response = client.responses.create(
                model=MODEL_NAME,
                instructions=system_prompt,
                input=original_profile,
                max_output_tokens=512,
                timeout=REQUEST_TIMEOUT,
            )

            text = response.output_text.strip()
            if not text:
                raise ValueError("빈 응답이 반환되었습니다.")

            return extract_response_text(text)

        except Exception as e:
            last_error = e
            wait_sec = min(2 ** attempt, 10)

            with print_lock:
                print(f"[WARN] attempt={attempt}/{MAX_RETRY} failed: {e} | wait={wait_sec}s")

            time.sleep(wait_sec)

    raise RuntimeError(f"생성 실패: {last_error}")

def worker(obj_id: int, profile: str, system_prompt: str):
    new_profile = generate_diverse_profile(system_prompt, profile)
    return obj_id, new_profile

def process_profiles_parallel(
    source_dict: dict,
    system_prompt: str,
    save_path: Path,
    id_key: str,
    profile_key: str = "profile",
):
    done_ids = read_existing_ids(save_path, id_key)

    tasks = []
    for id_str, profile in source_dict.items():
        obj_id = int(id_str)
        if obj_id not in done_ids:
            tasks.append((obj_id, profile))

    total = len(source_dict)
    remaining = len(tasks)
    skipped = len(done_ids)
    processed = 0
    failed = 0

    print(f"\n[START] {save_path.name}")
    print(f"전체={total}, 이미 저장됨={skipped}, 이번 실행 대상={remaining}")

    if remaining == 0:
        print(f"[DONE] {save_path.name} | 처리할 항목이 없습니다.")
        return

    with open(save_path, "a", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_id = {
                executor.submit(worker, obj_id, profile, system_prompt): obj_id
                for obj_id, profile in tasks
            }

            for idx, future in enumerate(as_completed(future_to_id), start=1):
                obj_id = future_to_id[future]

                try:
                    result_id, new_profile = future.result()

                    record = {
                        id_key: result_id,
                        profile_key: new_profile
                    }

                    with write_lock:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        f.flush()

                    processed += 1

                    if processed % 10 == 0 or processed == remaining:
                        with print_lock:
                            print(
                                f"[SAVE] {save_path.name} | "
                                f"새로 처리={processed}, 실패={failed}, "
                                f"완료={idx}/{remaining}"
                            )

                except Exception as e:
                    failed += 1
                    with print_lock:
                        print(f"[ERROR] {id_key}={obj_id} | {e}")

    print(
        f"[DONE] {save_path.name} | "
        f"새로 처리={processed}, 실패={failed}, "
        f"이미 저장됨={skipped}, 전체={total}"
    )

# =========================
# 데이터 로드
# =========================
user_profile = load_json(BASE_DIR / "user_profile.json")
item_profile = load_json(BASE_DIR / "item_profile.json")

user_system_prompt = load_text(Path("./generation/instruction/user_system_prompt_diverse.txt"))
item_system_prompt = load_text(Path("./generation/instruction/item_system_prompt_diverse.txt"))

# =========================
# 실행
# =========================
for diverse_no in range(DIVERSE_NUM):
    user_save_path = SAVE_DIR / f"diverse_user_profile_{diverse_no}.json"
    item_save_path = SAVE_DIR / f"diverse_item_profile_{diverse_no}.json"

    process_profiles_parallel(
        source_dict=user_profile,
        system_prompt=user_system_prompt,
        save_path=user_save_path,
        id_key="user_id",
    )

    process_profiles_parallel(
        source_dict=item_profile,
        system_prompt=item_system_prompt,
        save_path=item_save_path,
        id_key="item_id",
    )

print("\nDiverse profiles saved successfully.")