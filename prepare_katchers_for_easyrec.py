import pandas as pd
import numpy as np
import pickle
import os
from scipy.sparse import coo_matrix
from sklearn.model_selection import train_test_split

# ===== 설정 =====
DATA_PATH = "raw_data_katchers.txt"
SAVE_PATH = "data/katchers"

os.makedirs(SAVE_PATH, exist_ok=True)

# ===== 1. 데이터 읽기 =====
df = pd.read_csv(DATA_PATH, sep="\t")

# 필요한 컬럼만 사용
df = df[[
    "user_id",
    "product_id",
    "product_name",
    "category_name",
    "root_category_name",
    "attributes",
    "initial_paid_at"
]].copy()

df = df.dropna(subset=["user_id", "product_id"])
df = df.drop_duplicates(subset=["user_id", "product_id", "initial_paid_at"])

print("데이터 크기:", df.shape)

# ===== 2. ID 매핑 =====
user2id = {u: i for i, u in enumerate(df["user_id"].unique())}
item2id = {i: j for j, i in enumerate(df["product_id"].unique())}

df["uid"] = df["user_id"].map(user2id)
df["iid"] = df["product_id"].map(item2id)

num_users = len(user2id)
num_items = len(item2id)

print("유저 수:", num_users, "아이템 수:", num_items)

# ===== 3. train / val split =====
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# ===== 4. sparse matrix 생성 =====
def make_matrix(df):
    return coo_matrix(
        (np.ones(len(df)), (df["uid"], df["iid"])),
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

for _, row in df.iterrows():
    iid = str(item2id[row["product_id"]])

    text = f"{row['product_name']} {row['category_name']} {row['root_category_name']} {row['attributes']}"

    item_profiles[iid] = text

with open(os.path.join(SAVE_PATH, "item_profile.json"), "w", encoding="utf-8") as f:
    import json
    json.dump(item_profiles, f, ensure_ascii=False, indent=2)

print("item profile 저장 완료")

# ===== 6. user profile =====
user_profiles = {}

user_group = df.sort_values("initial_paid_at").groupby("uid")

for uid, group in user_group:
    item_texts = []
    for _, row in group.head(20).iterrows():  # 최근/초반 20개 정도만
        text = f"{row['product_name']} {row['category_name']} {row['root_category_name']} {row['attributes']}"
        item_texts.append(text)

    user_profiles[str(uid)] = " ; ".join(item_texts)

with open(os.path.join(SAVE_PATH, "user_profile.json"), "w", encoding="utf-8") as f:
    import json
    json.dump(user_profiles, f, ensure_ascii=False, indent=2)

print("user profile 저장 완료")