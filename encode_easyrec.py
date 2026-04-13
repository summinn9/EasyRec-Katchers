import os
import math
import json
import pickle
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from model import Easyrec


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='hkuds/easyrec-roberta-large', help='Model name')
parser.add_argument('--dataset', type=str, default='katchers')
parser.add_argument('--cuda', type=str, default='0')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--diverse_profile_num', type=int, default=1)
parser.add_argument('--save_root', type=str, default='./text_emb')
parser.add_argument('--skip_original', action='store_true')
parser.add_argument('--skip_diverse', action='store_true')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model_name_or_path = args.model
model_tag = model_name_or_path.split("/")[-1]
print("Model:", model_name_or_path)

config = AutoConfig.from_pretrained(model_name_or_path)
model = Easyrec.from_pretrained(
    model_name_or_path,
    from_tf=bool(".ckpt" in model_name_or_path),
    config=config,
).to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    use_fast=False,
)


def load_original_profiles(dataset: str):
    base_dir = Path(f'./data/{dataset}')

    with open(base_dir / 'user_profile.json', 'r', encoding='utf-8') as f:
        user_profile = json.load(f)

    with open(base_dir / 'item_profile.json', 'r', encoding='utf-8') as f:
        item_profile = json.load(f)

    user_profile_list = [user_profile[str(i)] for i in range(len(user_profile))]
    item_profile_list = [item_profile[str(i)] for i in range(len(item_profile))]

    return user_profile_list, item_profile_list


def load_diverse_profiles(dataset: str, diverse_no: int):
    base_dir = Path(f'./data/{dataset}/diverse_profile')

    user_profile = {}
    item_profile = {}

    user_path = base_dir / f'diverse_user_profile_{diverse_no}.json'
    item_path = base_dir / f'diverse_item_profile_{diverse_no}.json'

    with open(user_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            user_profile[int(data['user_id'])] = data['profile']

    with open(item_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            item_profile[int(data['item_id'])] = data['profile']

    user_profile_list = [user_profile[i] for i in range(len(user_profile))]
    item_profile_list = [item_profile[i] for i in range(len(item_profile))]

    return user_profile_list, item_profile_list


def save_batch_embedding(part_path: Path, emb_tensor: torch.Tensor):
    arr = emb_tensor.cpu().numpy()
    with open(part_path, 'wb') as f:
        pickle.dump(arr, f)


def load_batch_embedding(part_path: Path):
    with open(part_path, 'rb') as f:
        return pickle.load(f)


def combine_part_files(part_dir: Path, output_path: Path, expected_batches: int):
    parts = []
    for batch_idx in range(expected_batches):
        part_path = part_dir / f'batch_{batch_idx:05d}.pkl'
        if not part_path.exists():
            raise FileNotFoundError(f"Missing batch file: {part_path}")
        parts.append(load_batch_embedding(part_path))

    full_emb = np.concatenate(parts, axis=0)

    with open(output_path, 'wb') as f:
        pickle.dump(full_emb, f)

    print(f"[DONE] Final saved: {output_path}")


def encode_profiles_with_checkpoint(
    profiles,
    batch_size: int,
    desc: str,
    part_dir: Path,
):
    part_dir.mkdir(parents=True, exist_ok=True)

    n_batches = math.ceil(len(profiles) / batch_size)

    for batch_idx in tqdm(range(n_batches), desc=desc):
        part_path = part_dir / f'batch_{batch_idx:05d}.pkl'

        # 이미 저장된 배치면 건너뜀
        if part_path.exists():
            continue

        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, len(profiles))
        batch_profile = profiles[start:end]

        inputs = tokenizer(
            batch_profile,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            embeddings = model.encode(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )

        embeddings = F.normalize(
            embeddings.pooler_output.detach().float(),
            dim=-1
        )

        save_batch_embedding(part_path, embeddings)

        del inputs, embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return n_batches


def encode_and_save_separately(
    user_profiles,
    item_profiles,
    save_base_dir: Path,
    prefix: str,
    batch_size: int,
):
    save_base_dir.mkdir(parents=True, exist_ok=True)

    # user
    user_part_dir = save_base_dir / f'user_{prefix}_parts'
    user_final_path = save_base_dir / f'user_{prefix}.pkl'

    print(f"\n=== Encoding USER: {prefix} ===")
    user_batches = encode_profiles_with_checkpoint(
        profiles=user_profiles,
        batch_size=batch_size,
        desc=f'user_{prefix}',
        part_dir=user_part_dir,
    )

    if not user_final_path.exists():
        combine_part_files(user_part_dir, user_final_path, user_batches)
    else:
        print(f"[SKIP] Final already exists: {user_final_path}")

    # item
    item_part_dir = save_base_dir / f'item_{prefix}_parts'
    item_final_path = save_base_dir / f'item_{prefix}.pkl'

    print(f"\n=== Encoding ITEM: {prefix} ===")
    item_batches = encode_profiles_with_checkpoint(
        profiles=item_profiles,
        batch_size=batch_size,
        desc=f'item_{prefix}',
        part_dir=item_part_dir,
    )

    if not item_final_path.exists():
        combine_part_files(item_part_dir, item_final_path, item_batches)
    else:
        print(f"[SKIP] Final already exists: {item_final_path}")


def main():
    dataset = args.dataset
    save_root = Path(args.save_root)
    save_path = save_root / dataset
    save_path.mkdir(parents=True, exist_ok=True)

    # 1) original generated profiles
    if not args.skip_original:
        print("\n=== Encoding original generated profiles ===")
        orig_user_profiles, orig_item_profiles = load_original_profiles(dataset)

        encode_and_save_separately(
            user_profiles=orig_user_profiles,
            item_profiles=orig_item_profiles,
            save_base_dir=save_path,
            prefix=model_tag,
            batch_size=args.batch_size,
        )

    # 2) diversified profiles
    if not args.skip_diverse:
        diverse_save_dir = save_path / 'diverse_profile'
        diverse_save_dir.mkdir(parents=True, exist_ok=True)

        for diverse_no in range(args.diverse_profile_num):
            print(f"\n=== Encoding diverse profiles #{diverse_no} ===")
            div_user_profiles, div_item_profiles = load_diverse_profiles(dataset, diverse_no)

            encode_and_save_separately(
                user_profiles=div_user_profiles,
                item_profiles=div_item_profiles,
                save_base_dir=diverse_save_dir,
                prefix=f'{model_tag}_{diverse_no}',
                batch_size=args.batch_size,
            )

    print("\nAll embeddings saved successfully.")


if __name__ == "__main__":
    main()