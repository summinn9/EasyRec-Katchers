import pandas as pd
import numpy as np
import pickle
import os
import re
import json
from collections import OrderedDict
from scipy.sparse import coo_matrix
from sklearn.model_selection import train_test_split

# ===== 설정 =====
DATA_PATH = "raw_data_katchers.txt"
SAVE_PATH = "data/katchers"

os.makedirs(SAVE_PATH, exist_ok=True)

# ===== 정제 패턴 =====

BRACKET_PATTERN = re.compile(r"\[[^\]]+\]")
META_PATTERN = re.compile(r"(선택|옵션|구성|색상|용량|부위|개수)\|")
NULL_PATTERN = re.compile(r"\b(nan|none|null)\b", re.IGNORECASE)

UNIT_PATTERN = re.compile(
    r"\b\d+(\.\d+)?\s*(kg|g|mg|ml|l|매|롤|개입|개|팩|봉|박스|세트|인분|미|회분|종|호|과)\b",
    re.IGNORECASE
)

PLUS_PATTERN = re.compile(r"\b\d+\s*\+\s*\d+\b")

RANGE_PATTERN = re.compile(
    r"\b\d+\s*[-~]\s*\d+(\.\d+)?\s*(kg|g|mg|ml|l|개|팩|인분|호|과)?\b",
    re.IGNORECASE
)

NUMBER_PATTERN = re.compile(r"\b\d+\b")

MARKETING_WORDS = [
    "미친가성비", "가성비", "고급스러운", "프리미엄",
    "최고의", "특가", "행사", "추천", "필수품",
    "추억의", "옛날", "새콤달달", "겉바속촉",
    "붓기만 하면", "간편하게", "든든한",
    "풍미가득", "대용량", "인기", "히트", "베스트",
    "첫 출하", "갓 수확", "국산", "한번에 발송",
    "출고 가능", "이후 출고 가능", "증정", "총",
    "한박스", "소포장", "신선한", "명품", "급이 다른",
    "꼼꼼한", "상쾌함", "물놀이", "안전하고", "깨끗한"
]
MARKETING_PATTERN = re.compile("|".join(map(re.escape, MARKETING_WORDS)))

SPECIAL_PATTERN = re.compile(r"[|/;,:\[\]\(\)\+\-_=~!?.&★%]")
MULTISPACE_PATTERN = re.compile(r"\s+")


def clean_text_for_llm(text):
    if pd.isna(text):
        return ""

    text = str(text).lower().strip()

    if not text:
        return ""

    text = BRACKET_PATTERN.sub(" ", text)
    text = META_PATTERN.sub(" ", text)
    text = NULL_PATTERN.sub(" ", text)

    text = PLUS_PATTERN.sub(" ", text)
    text = RANGE_PATTERN.sub(" ", text)
    text = UNIT_PATTERN.sub(" ", text)

    text = MARKETING_PATTERN.sub(" ", text)
    text = SPECIAL_PATTERN.sub(" ", text)
    text = NUMBER_PATTERN.sub(" ", text)

    text = MULTISPACE_PATTERN.sub(" ", text).strip()
    return text


def build_item_text_for_llm(row):
    parts = [
        clean_text_for_llm(row.get("product_name", "")),
        clean_text_for_llm(row.get("category_name", "")),
        clean_text_for_llm(row.get("root_category_name", "")),
        clean_text_for_llm(row.get("attributes", "")),
    ]
    parts = [p for p in parts if p]
    return " ".join(parts).strip()


def get_item_text_with_fallback(row):
    text = build_item_text_for_llm(row)

    if not text:
        text = clean_text_for_llm(row.get("product_name", ""))

    if not text:
        text = clean_text_for_llm(row.get("category_name", ""))

    if not text:
        text = clean_text_for_llm(row.get("root_category_name", ""))

    return text.strip()


# ===== 1. 데이터 읽기 =====
df = pd.read_csv(DATA_PATH, sep="\t")

df = df[[
    "user_id",
    "order_id",
    "order_code",
    "product_id",
    "product_name",
    "category_name",
    "root_category_name",
    "attributes",
    "initial_paid_at",
    "day"
]].copy()

df = df.dropna(subset=["user_id", "product_id"])
df = df.drop_duplicates(subset=["user_id", "product_id", "order_id", "initial_paid_at"])

df["initial_paid_at"] = pd.to_datetime(df["initial_paid_at"], unit="ms", errors="coerce")
print("데이터 크기:", df.shape)

# ===== 2. ID 매핑 =====
user2id = {u: i for i, u in enumerate(df["user_id"].unique())}
item2id = {i: j for j, i in enumerate(df["product_id"].unique())}

df["uid"] = df["user_id"].map(user2id)
df["iid"] = df["product_id"].map(item2id)

num_users = len(user2id)
num_items = len(item2id)

print("유저 수:", num_users, "아이템 수:", num_items)

# ===== 3. train / val / test split =====
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

print("train 크기:", train_df.shape)
print("val 크기:", val_df.shape)
print("test 크기:", test_df.shape)

# ===== 4. sparse matrix 생성 =====
def make_matrix(dataframe):
    return coo_matrix(
        (np.ones(len(dataframe)), (dataframe["uid"], dataframe["iid"])),
        shape=(num_users, num_items)
    )

trn_mat = make_matrix(train_df)
val_mat = make_matrix(val_df)
tst_mat = make_matrix(test_df)

with open(os.path.join(SAVE_PATH, "trn_mat.pkl"), "wb") as f:
    pickle.dump(trn_mat, f)

with open(os.path.join(SAVE_PATH, "val_mat.pkl"), "wb") as f:
    pickle.dump(val_mat, f)

with open(os.path.join(SAVE_PATH, "tst_mat.pkl"), "wb") as f:
    pickle.dump(tst_mat, f)

print("matrix 저장 완료")

# ===== 5. item profile =====
item_profiles = {}

item_df = (
    df.sort_values("initial_paid_at", ascending=False)
      .drop_duplicates(subset=["product_id"])
)

for _, row in item_df.iterrows():
    iid = str(item2id[row["product_id"]])

    text = get_item_text_with_fallback(row)

    if not text:
        text = "unknown item"

    item_profiles[iid] = text

with open(os.path.join(SAVE_PATH, "item_profile.json"), "w", encoding="utf-8") as f:
    json.dump(item_profiles, f, ensure_ascii=False, indent=2)

print("item profile 저장 완료")

# ===== 6. user profile: order-aware version =====
from collections import OrderedDict, defaultdict

user_profiles = {}

train_user_group = (
    train_df.sort_values("initial_paid_at", ascending=False)
            .groupby("uid")
)

for uid, group in train_user_group:
    product_count = defaultdict(int)
    profile_parts = []

    # 최근 주문 순서 유지
    order_group = (
        group.sort_values("initial_paid_at", ascending=False)
             .groupby("order_id", sort=False)
    )

    for order_id, order_rows in order_group:
        order_item_map = OrderedDict()

        for _, row in order_rows.iterrows():
            pid = row["product_id"]
            text = get_item_text_with_fallback(row)

            if not text:
                continue

            product_count[pid] += 1

            # 같은 주문 안에서 같은 상품 중복 방지
            if pid not in order_item_map:
                order_item_map[pid] = text

        if not order_item_map:
            continue

        order_item_texts = list(order_item_map.values())

        MAX_ITEMS_PER_ORDER = 5
        order_item_texts = order_item_texts[:MAX_ITEMS_PER_ORDER]

        # 같은 주문에서 2개 이상 구매한 경우 = 동시구매 정보 반영
        if len(order_item_texts) >= 2:
            part = "Recent order with items bought together: " + " | ".join(order_item_texts)
        else:
            part = "Recent single purchase item: " + order_item_texts[0]

        profile_parts.append(part)

        # 너무 길어지는 것 방지: 최근 주문 기준 최대 20개
        if len(profile_parts) >= 20:
            break

    # 반복구매 정보 추가
    repeat_parts = []
    for pid, count in product_count.items():
        if count >= 2:
            item_text = item_profiles[str(item2id[pid])]
            repeat_parts.append(f"Repeated purchase item: {item_text} repeat_purchase_{count}")

    final_parts = profile_parts + repeat_parts[:5]

    MAX_REPEAT_ITEMS = 5
    final_parts = profile_parts + repeat_parts[:MAX_REPEAT_ITEMS]

    if final_parts:
        user_profiles[str(uid)] = " ; ".join(final_parts)
    else:
        user_profiles[str(uid)] = "no purchase profile"

for uid in range(num_users):
    user_profiles.setdefault(str(uid), "no purchase profile")

with open(os.path.join(SAVE_PATH, "user_profile.json"), "w", encoding="utf-8") as f:
    json.dump(user_profiles, f, ensure_ascii=False, indent=2)

print("order-aware user profile 저장 완료")

# ===== 7. 통계 확인 =====
empty_item_count = sum(1 for v in item_profiles.values() if not str(v).strip())
empty_user_count = sum(1 for v in user_profiles.values() if not str(v).strip())

fallback_user_count = sum(1 for v in user_profiles.values() if v == "no purchase profile")
fallback_item_count = sum(1 for v in item_profiles.values() if v == "unknown item")

repeat_tag_user_count = sum(
    1 for v in user_profiles.values() if "repeat_purchase_" in v
)

print("\n[통계]")
print("빈 item profile 수:", empty_item_count)
print("빈 user profile 수:", empty_user_count)
print("fallback item 수:", fallback_item_count)
print("fallback user 수:", fallback_user_count)
print("반복구매 태그 포함 user 수:", repeat_tag_user_count)

# ===== 8. 샘플 출력 =====
print("\n[item profile sample]")
for k in list(item_profiles.keys())[:5]:
    print(f"{k}: {item_profiles[k]}")

print("\n[user profile sample]")
for k in list(user_profiles.keys())[:5]:
    print(f"{k}: {user_profiles[k]}")