# -*- coding: utf-8 -*-
"""
ItemCF with RecBole-style TopK metrics:
recall@K, precision@K, hit@K, mrr@K, ndcg@K

- LOO by timestamp per user: last->test, second last->valid, rest->train
- Eval modes:
  * full: rank against all unseen items
  * sampled: rank against 1 positive + N negatives (default 99)

Key fix:
- "adjusted" similarity uses TopK-neighbor graph (NO full item-item matrix),
  so it won't OOM on ml-latest-small.
"""

import os
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# ===================== 配置区 =====================
RATINGS_PATH = "ratings.csv"

SIM_METHOD = "adjusted"   # "adjusted" or "cosine"
K_NEIGHBORS = 20          # ItemCF 的相似物品数（也是相似度图的 TopK）
TOPK = 10                 # 输出指标的 K（比如 @10）

EVAL_MODE = "sampled"     # "sampled" or "full"
NEG_SAMPLE = 99           # sampled 模式下每个正样本配多少负样本
MAX_USERS = 1000          # 评估最多多少用户（防止太慢），None 表示全量
SEED = 42
# ================================================


def make_results_dir():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"results_{ts}"
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def leave_one_out_split(ratings: pd.DataFrame):
    df = ratings.sort_values(["userId", "timestamp"]).reset_index(drop=True)

    test_idx = df.groupby("userId").tail(1).index
    df_wo_test = df.drop(index=test_idx)
    valid_idx = df_wo_test.groupby("userId").tail(1).index
    train_idx = df.index.difference(test_idx.union(valid_idx))

    train_df = df.loc[train_idx].reset_index(drop=True)
    valid_df = df.loc[valid_idx].reset_index(drop=True)
    test_df  = df.loc[test_idx].reset_index(drop=True)
    return train_df, valid_df, test_df


def build_user_seen(train_df: pd.DataFrame):
    """每个用户在训练集中看过哪些 item（用于过滤负样本/全量候选）"""
    seen = {}
    for u, g in train_df.groupby("userId")["movieId"]:
        seen[int(u)] = set(map(int, g.values))
    return seen


class ItemCFRecommender:
    def __init__(self, similarity_method="adjusted", K=20):
        self.similarity_method = similarity_method
        self.K = K
        self.user_item = None  # DataFrame: user x item rating (0=missing)
        self.global_mean = 3.0

        # TopK neighbor graph (for adjusted)
        # item_id -> (neighbor_item_ids ndarray, neighbor_sims ndarray)
        self.item_topk = {}

        # Full sim matrix (only for cosine option; still might be big)
        self.item_sim = None

    def fit(self, train_df: pd.DataFrame):
        self.global_mean = float(train_df["rating"].mean())

        self.user_item = train_df.pivot_table(
            index="userId", columns="movieId", values="rating", fill_value=0
        )

        if self.similarity_method == "cosine":
            # 注意：cosine 全量矩阵也可能大（n_item^2），latest-small 可能会占内存
            sim = cosine_similarity(self.user_item.T)
            self.item_sim = pd.DataFrame(sim, index=self.user_item.columns, columns=self.user_item.columns)
            return self

        # adjusted: mean-center per user, then compute TopK cosine neighbors (no full matrix)
        ui = self.user_item.replace(0, np.nan)
        user_mean = ui.mean(axis=1).fillna(self.global_mean)
        centered = self.user_item.sub(user_mean, axis=0).fillna(0).astype(np.float32)

        item_ids = self.user_item.columns.to_numpy(dtype=int)
        X = centered.T.to_numpy(dtype=np.float32)  # (n_item, n_user)

        # 用 cosine_similarity 分块算 TopK（不保存全矩阵）
        # 块大小可调：越大越快但越吃内存
        block = 512
        n_items = X.shape[0]

        for start in range(0, n_items, block):
            end = min(start + block, n_items)
            sims_block = cosine_similarity(X[start:end], X)  # (block, n_items)

            for i in range(end - start):
                row = sims_block[i]
                idx_self = start + i
                row[idx_self] = -1.0  # 排除自己

                # 取 TopK
                if self.K < len(row):
                    top_idx = np.argpartition(-row, self.K)[:self.K]
                else:
                    top_idx = np.argsort(-row)

                top_idx = top_idx[np.argsort(-row[top_idx])]
                top_sims = row[top_idx]

                # 只保留正相似度（可选：更稳）
                mask = top_sims > 0
                nbrs = item_ids[top_idx][mask]
                nbr_sims = top_sims[mask].astype(np.float32)

                self.item_topk[int(item_ids[idx_self])] = (nbrs, nbr_sims)

        return self

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        if user_id not in self.user_item.index:
            return self.global_mean

        user_vec = self.user_item.loc[user_id]
        rated = user_vec[user_vec > 0]
        if len(rated) == 0:
            return self.global_mean

        if movie_id not in self.user_item.columns:
            return float(rated.mean())

        # cosine 全量矩阵分支
        if self.similarity_method == "cosine":
            sims = self.item_sim[movie_id].drop(index=movie_id, errors="ignore")
            topk = sims.sort_values(ascending=False).head(self.K)
            num = 0.0
            den = 0.0
            for m, s in topk.items():
                r = user_vec.get(m, 0)
                if r > 0 and s > 0:
                    num += s * r
                    den += s
            return num / den if den > 0 else float(rated.mean())

        # adjusted TopK neighbor graph 分支
        nbrs, nbr_sims = self.item_topk.get(int(movie_id), (None, None))
        if nbrs is None or len(nbrs) == 0:
            return float(rated.mean())

        num = 0.0
        den = 0.0
        for m, s in zip(nbrs, nbr_sims):
            r = float(user_vec.get(m, 0))
            if r > 0:
                num += float(s) * r
                den += float(s)
        return num / den if den > 0 else float(rated.mean())

    def score_items(self, user_id: int, item_ids):
        return np.array([self.predict_rating(user_id, int(i)) for i in item_ids], dtype=np.float32)


def _metrics_for_one(ranked_items, gt_item, topk: int):
    top = ranked_items[:topk]
    hit = 1.0 if gt_item in top else 0.0
    precision = hit / topk
    recall = hit

    if hit:
        rank = top.index(gt_item) + 1
        mrr = 1.0 / rank
        ndcg = 1.0 / np.log2(rank + 1)
    else:
        mrr = 0.0
        ndcg = 0.0
    return recall, precision, hit, mrr, ndcg


def evaluate_topk(model: ItemCFRecommender,
                  train_seen: dict,
                  eval_df: pd.DataFrame,
                  topk=10,
                  mode="sampled",
                  neg_sample=99,
                  max_users=1000,
                  seed=42):
    rng = np.random.default_rng(seed)
    eval_pairs = eval_df[["userId", "movieId"]].astype(int).values.tolist()

    if max_users is not None and len(eval_pairs) > max_users:
        rng.shuffle(eval_pairs)
        eval_pairs = eval_pairs[:max_users]

    all_items = model.user_item.columns.to_numpy(dtype=int)

    sums = {"recall": 0.0, "precision": 0.0, "hit": 0.0, "mrr": 0.0, "ndcg": 0.0}
    n = 0

    for uid, gt in eval_pairs:
        if uid not in model.user_item.index:
            continue

        seen = set(train_seen.get(int(uid), set()))
        seen.add(int(gt))

        if mode == "full":
            candidates = [i for i in all_items.tolist() if i not in seen] + [int(gt)]
        else:
            pool = [i for i in all_items.tolist() if i not in seen]
            if len(pool) <= neg_sample:
                negs = pool
            else:
                negs = rng.choice(pool, size=neg_sample, replace=False).tolist()
            candidates = negs + [int(gt)]

        scores = model.score_items(uid, candidates)
        order = np.argsort(-scores)
        ranked = [candidates[i] for i in order]

        recall, precision, hit, mrr, ndcg = _metrics_for_one(ranked, int(gt), topk)
        sums["recall"] += recall
        sums["precision"] += precision
        sums["hit"] += hit
        sums["mrr"] += mrr
        sums["ndcg"] += ndcg
        n += 1

    if n == 0:
        return {f"recall@{topk}": 0.0, f"precision@{topk}": 0.0, f"hit@{topk}": 0.0,
                f"mrr@{topk}": 0.0, f"ndcg@{topk}": 0.0, "users": 0}

    return {
        f"recall@{topk}": sums["recall"] / n,
        f"precision@{topk}": sums["precision"] / n,
        f"hit@{topk}": sums["hit"] / n,
        f"mrr@{topk}": sums["mrr"] / n,
        f"ndcg@{topk}": sums["ndcg"] / n,
        "users": n
    }


def main():
    out_dir = make_results_dir()
    ratings = pd.read_csv(RATINGS_PATH)

    train_df, valid_df, test_df = leave_one_out_split(ratings)
    train_seen = build_user_seen(train_df)

    model = ItemCFRecommender(similarity_method=SIM_METHOD, K=K_NEIGHBORS).fit(train_df)

    valid_metrics = evaluate_topk(
        model, train_seen, valid_df,
        topk=TOPK, mode=EVAL_MODE, neg_sample=NEG_SAMPLE,
        max_users=MAX_USERS, seed=SEED
    )
    test_metrics = evaluate_topk(
        model, train_seen, test_df,
        topk=TOPK, mode=EVAL_MODE, neg_sample=NEG_SAMPLE,
        max_users=MAX_USERS, seed=SEED
    )

    print("best valid :", {k: round(v, 4) for k, v in valid_metrics.items() if k != "users"})
    print("test result:", {k: round(v, 4) for k, v in test_metrics.items() if k != "users"})

    with open(os.path.join(out_dir, "recbole_style_metrics.txt"), "w", encoding="utf-8") as f:
        f.write("VALID:\n")
        for k, v in valid_metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\nTEST:\n")
        for k, v in test_metrics.items():
            f.write(f"{k}: {v}\n")

    print(f"Saved to: {out_dir}/recbole_style_metrics.txt")


if __name__ == "__main__":
    main()
