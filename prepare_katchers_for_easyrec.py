import pandas as pd
import numpy as np
import pickle
import os
import re
import json
from collections import OrderedDict, defaultdict
from scipy.sparse import coo_matrix
from sklearn.model_selection import GroupShuffleSplit

# ===== 설정 =====
DATA_PATH = "raw_data_katchers.txt"
SAVE_PATH = "data/katchers"
SEED = 42

os.makedirs(SAVE_PATH, exist_ok=True)

# ===== 정제 패턴 =====
BRACKET_PATTERN = re.compile(r"\[[^\]]+\]")
META_PATTERN = re.compile(r"(선택|옵션|구성|색상|용량|부위|개수)\|")
NULL_PATTERN = re.compile(r"\b(nan|none|null)\b", re.IGNORECASE)
UNIT_PATTERN = re.compile(
    r"\b\d+(\.\d)?\s*(kg|g|mg|ml|l|매|롤|개입|개|팩|봉|박스|세트|인분|미|회분|종|호|과)\b",
    re.IGNORECASE
)
PLUS_PATTERN = re.compile(r"\b\d+\s*\+\s*\d+\b")
RANGE_PATTERN = re.compile(
    r"\b\d+\s*[-~]\s*\d+(\.\d+)?\s*(kg|g|mg|ml|l|개|팩|인분|호|과)?\b",
    re.IGNORECASE
)
NUMBER_PATTERN = re.compile(r"\b\d+\b")
SPECIAL_PATTERN = re.compile(r"[|/;,:\[\]\(\)\+\-_=~!?.&★%]")
MULTISPACE_PATTERN = re.compile(r"\s+")

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

print("[필터링 전]")
print("데이터 크기:", df.shape)
print("유저 수:", df["user_id"].nunique())
print("아이템 수:", df["product_id"].nunique())


# ===== 2. 유저 필터링 =====
user_stats = df.groupby("user_id").agg(
    purchase_days=("day", "nunique"),
    purchase_count=("product_id", "count")
).reset_index()

valid_users = user_stats[
    (user_stats["purchase_days"] >= 2) &
    (user_stats["purchase_count"] < 68)
]["user_id"]

df = df[df["user_id"].isin(valid_users)].copy()

print("\n[필터링 후]")
print("데이터 크기:", df.shape)
print("유저 수:", df["user_id"].nunique())
print("아이템 수:", df["product_id"].nunique())


# ===== 3. ID 매핑 =====
user2id = {u: i for i, u in enumerate(df["user_id"].unique())}
item2id = {p: i for i, p in enumerate(df["product_id"].unique())}

df["uid"] = df["user_id"].map(user2id)
df["iid"] = df["product_id"].map(item2id)

num_users = len(user2id)
num_items = len(item2id)

print("\n[ID 매핑 후]")
print("유저 수:", num_users)
print("아이템 수:", num_items)


# ===== 4. train / test split =====
# 깃허브 코드 방식에 맞춰 user_id 기준 group split
gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=SEED)
train_idx, test_idx = next(gss.split(df, groups=df["user_id"]))

train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()

print("\n[Split 결과]")
print("train 크기:", train_df.shape)
print("test 크기:", test_df.shape)
print("train 유저 수:", train_df["uid"].nunique())
print("test 유저 수:", test_df["uid"].nunique())
print("train 아이템 수:", train_df["iid"].nunique())
print("test 아이템 수:", test_df["iid"].nunique())


# ===== 5. sparse matrix 생성 =====
def make_matrix(dataframe):
    return coo_matrix(
        (np.ones(len(dataframe)), (dataframe["uid"], dataframe["iid"])),
        shape=(num_users, num_items)
    )

trn_mat = make_matrix(train_df)
tst_mat = make_matrix(test_df)

# 기존 파일 덮어쓰기
with open(os.path.join(SAVE_PATH, "trn_mat.pkl"), "wb") as f:
    pickle.dump(trn_mat, f)

with open(os.path.join(SAVE_PATH, "tst_mat.pkl"), "wb") as f:
    pickle.dump(tst_mat, f)

# EasyRec이 val_mat 필요로 하면 형식 맞추기용으로 저장
with open(os.path.join(SAVE_PATH, "val_mat.pkl"), "wb") as f:
    pickle.dump(tst_mat, f)

print("\n기존 matrix 파일 덮어쓰기 완료")


# ===== 6. split csv / mapping 저장 =====
df.to_csv(os.path.join(SAVE_PATH, "filtered_all.csv"), index=False)
train_df.to_csv(os.path.join(SAVE_PATH, "train_df.csv"), index=False)
test_df.to_csv(os.path.join(SAVE_PATH, "test_df.csv"), index=False)

with open(os.path.join(SAVE_PATH, "user2id.json"), "w", encoding="utf-8") as f:
    json.dump({str(k): int(v) for k, v in user2id.items()}, f, ensure_ascii=False, indent=2)

with open(os.path.join(SAVE_PATH, "item2id.json"), "w", encoding="utf-8") as f:
    json.dump({str(k): int(v) for k, v in item2id.items()}, f, ensure_ascii=False, indent=2)

print("split csv / mapping 저장 완료")


# ===== 7. item_profile.json 덮어쓰기 =====
# 주의: 이건 LLM 결과가 아니라 LLM 입력용 raw item profile임

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

print("기존 item_profile.json 덮어쓰기 완료")


# ===== 8. user_profile.json 덮어쓰기 =====
# 중요: train_df 기준으로만 생성

user_profiles = {}

train_user_group = (
    train_df.sort_values("initial_paid_at", ascending=False)
            .groupby("uid")
)

for uid, group in train_user_group:
    product_count = defaultdict(int)
    profile_parts = []

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

            if pid not in order_item_map:
                order_item_map[pid] = text

        if not order_item_map:
            continue

        order_item_texts = list(order_item_map.values())[:5]

        if len(order_item_texts) >= 2:
            part = "Recent order with items bought together: " + " | ".join(order_item_texts)
        else:
            part = "Recent single purchase item: " + order_item_texts[0]

        profile_parts.append(part)

        if len(profile_parts) >= 20:
            break

    repeat_parts = []
    for pid, count in product_count.items():
        if count >= 2:
            mapped_iid = str(item2id[pid])
            item_text = item_profiles.get(mapped_iid, "unknown item")
            repeat_parts.append(f"Repeated purchase item: {item_text} repeat_purchase_{count}")

    final_parts = profile_parts + repeat_parts[:5]

    if final_parts:
        user_profiles[str(uid)] = " ; ".join(final_parts)
    else:
        user_profiles[str(uid)] = "no purchase profile"

for uid in range(num_users):
    user_profiles.setdefault(str(uid), "no purchase profile")

with open(os.path.join(SAVE_PATH, "user_profile.json"), "w", encoding="utf-8") as f:
    json.dump(user_profiles, f, ensure_ascii=False, indent=2)

print("기존 user_profile.json 덮어쓰기 완료")


# ===== 9. 통계 확인 =====
print("\n[통계]")
print("item profile 수:", len(item_profiles))
print("user profile 수:", len(user_profiles))
print("fallback item 수:", sum(1 for v in item_profiles.values() if v == "unknown item"))
print("fallback user 수:", sum(1 for v in user_profiles.values() if v == "no purchase profile"))
print("반복구매 태그 포함 user 수:", sum(1 for v in user_profiles.values() if "repeat_purchase_" in v))

print("\n[item profile sample]")
for k in list(item_profiles.keys())[:5]:
    print(f"{k}: {item_profiles[k]}")

print("\n[user profile sample]")
for k in list(user_profiles.keys())[:3]:
    print(f"{k}: {user_profiles[k]}")