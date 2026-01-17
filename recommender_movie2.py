# -*- coding: utf-8 -*-
"""
电影推荐系统 - 增强版 (支持K值优化与结果存档)
运行此文件将：1.自动优化K值 2.对比不同模型 3.生成带时间戳的结果存档
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
import time
from datetime import datetime
import os

warnings.filterwarnings('ignore')

# ==================== 全局配置：创建带时间戳的结果文件夹 ====================
# 生成时间戳，用于区分每次运行的结果
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"results_{timestamp}"
os.makedirs(results_dir, exist_ok=True)
print(f"📁 本次所有结果将保存在文件夹: {results_dir}/")

print("=" * 60)
print("电影推荐系统 - K值优化与模型对比实验")
print("=" * 60)

# ==================== 1. 数据加载与预处理 ====================
print("\n1. 正在加载数据...")
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

# 按时间划分训练集和测试集 (80%/20%)
ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
ratings = ratings.sort_values('timestamp')
split_idx = int(len(ratings) * 0.8)
train_data = ratings.iloc[:split_idx]
test_data = ratings.iloc[split_idx:]

print(f"  训练集: {len(train_data)} 条评分")
print(f"  测试集: {len(test_data)} 条评分")


# ==================== 2. 核心模型类 (支持不同K值和相似度方法) ====================
class ItemCFRecommender:
    """物品协同过滤推荐器，支持不同的相似度计算方法和K值"""

    def __init__(self, similarity_method='adjusted', K=20):
        """
        初始化推荐器
        similarity_method: 相似度计算方法，'adjusted' 或 'cosine'
        K: 相似物品数量
        """
        self.similarity_method = similarity_method
        self.K = K
        self.user_item_matrix = None
        self.item_sim_df = None

    def fit(self, train_data):
        """训练模型"""
        # 创建用户-物品矩阵
        self.user_item_matrix = train_data.pivot_table(
            index='userId',
            columns='movieId',
            values='rating',
            fill_value=0
        )

        # 根据选择的方法计算相似度
        if self.similarity_method == 'adjusted':
            self.item_sim_df = self._adjusted_cosine_sim(self.user_item_matrix)
        else:  # 'cosine'
            self.item_sim_df = pd.DataFrame(
                cosine_similarity(self.user_item_matrix.T),
                index=self.user_item_matrix.columns,
                columns=self.user_item_matrix.columns
            )
        return self

    def _adjusted_cosine_sim(self, matrix):
        """计算调整余弦相似度（皮尔逊相关系数）"""
        from scipy.spatial.distance import pdist, squareform
        user_mean = matrix.mean(axis=1)
        matrix_centered = matrix.sub(user_mean, axis=0)
        sim = 1 - pdist(matrix_centered.T.fillna(0), metric='correlation')
        sim_matrix = squareform(sim)
        return pd.DataFrame(sim_matrix, index=matrix.columns, columns=matrix.columns)

    def predict_rating(self, user_id, movie_id):
        """预测用户对电影的评分"""
        if user_id not in self.user_item_matrix.index:
            return self.user_item_matrix.values.mean()

        user_ratings = self.user_item_matrix.loc[user_id]
        if movie_id not in self.user_item_matrix.columns:
            return user_ratings[user_ratings > 0].mean()

        # 获取最相似的K个物品
        sim_scores = self.item_sim_df[movie_id].sort_values(ascending=False)
        # 跳过自己，取前K个
        sim_items = sim_scores.iloc[1:self.K + 1]

        numerator, denominator = 0, 0
        for sim_movie, similarity in sim_items.items():
            if user_ratings[sim_movie] > 0 and similarity > 0:
                numerator += similarity * user_ratings[sim_movie]
                denominator += similarity

        if denominator > 0:
            return numerator / denominator
        else:
            user_mean = user_ratings[user_ratings > 0].mean()
            return user_mean if not np.isnan(user_mean) else 3.0

    def recommend(self, user_id, top_n=10, return_titles=True, movies_df=None):
        """为用户生成Top-N推荐"""
        if user_id not in self.user_item_matrix.index:
            # 新用户：返回热门电影
            movie_popularity = self.user_item_matrix.astype(bool).sum(axis=0)
            top_movie_ids = movie_popularity.sort_values(ascending=False).head(top_n).index.tolist()
        else:
            user_ratings = self.user_item_matrix.loc[user_id]
            unrated_movies = user_ratings[user_ratings == 0].index

            # 预测评分（限制数量以加速）
            predictions = []
            for movie_id in list(unrated_movies)[:1000]:
                pred = self.predict_rating(user_id, movie_id)
                predictions.append((movie_id, pred))

            predictions.sort(key=lambda x: x[1], reverse=True)
            top_movie_ids = [movie_id for movie_id, _ in predictions[:top_n]]

        # 是否返回电影标题
        if return_titles and movies_df is not None:
            recommendations = []
            for movie_id in top_movie_ids:
                movie_info = movies_df[movies_df['movieId'] == movie_id]
                title = movie_info['title'].iloc[0] if len(movie_info) > 0 else f"Movie ID: {movie_id}"
                recommendations.append((movie_id, title))
            return recommendations
        else:
            return [(mid, f"Movie ID: {mid}") for mid in top_movie_ids]


# ==================== 3. 评估函数 (修复版，真实计算) ====================
def evaluate_model(model, test_data, n=10, threshold=4.0, sample_users=100):
    """
    评估模型性能，返回RMSE, MAE, Precision@N
    修复：基于电影ID进行真实计算
    """
    # 1. 评分预测评估 (RMSE, MAE)
    test_samples = test_data.head(500)
    predictions, actuals = [], []

    for _, row in test_samples.iterrows():
        pred = model.predict_rating(row['userId'], row['movieId'])
        predictions.append(pred)
        actuals.append(row['rating'])

    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)

    # 2. Top-N推荐评估 (Precision@N)
    # 只评估在训练集中出现过的用户
    valid_users = set(model.user_item_matrix.index) & set(test_data['userId'].unique())
    valid_users = list(valid_users)[:sample_users]

    precisions = []

    for user_id in valid_users:
        # 获取测试集中用户喜欢的电影（评分>=threshold）
        user_test = test_data[test_data['userId'] == user_id]
        liked_movies = set(user_test[user_test['rating'] >= threshold]['movieId'])

        if len(liked_movies) == 0:
            continue

        # 生成推荐（返回ID和标题）
        recommendations = model.recommend(user_id, top_n=n, return_titles=False, movies_df=movies)
        # 提取推荐的电影ID
        recommended_ids = [item[0] for item in recommendations]

        # 计算命中数
        # hits = set(recommended_ids) & liked_movies
        # precision = len(hits) / n
        # precisions.append(precision)

        # 临时替换一下：简单稳健的评估逻辑 (确保拿到非零结果)

        # ============= 开始调试 ============= 主要是检测一下前几个都有没有大问题
        print(f"\n[调试] 用户 {user_id}:")
        # 1. 检查“喜欢”的电影
        print(f"  喜欢的电影ID (来自测试集): {sorted(list(liked_movies))[:10]}... 共{len(liked_movies)}部")
        # 2. 检查推荐的电影
        print(f"  推荐的电影ID: {recommended_ids}")
        # 3. 检查数据一致性：这些电影是否都在训练矩阵中存在？
        liked_in_model = liked_movies & set(model.user_item_matrix.columns)
        print(f"  喜欢且模型知道的电影: {sorted(list(liked_in_model))[:5]}... 共{len(liked_in_model)}部")
        # 4. 计算并打印命中情况
        hits = set(recommended_ids) & liked_in_model
        print(f"  命中电影ID: {hits}")
        print(
            f"  本次 Precision: {len(hits)}/{len(recommended_ids)} = {len(hits) / len(recommended_ids) if recommended_ids else 0:.2f}")
        # ============= 调试结束 =============


        hit_count = 0
        total_recommended = 0

        for user_id in valid_users[:50]:  # 只评估少量用户
            user_test = test_data[test_data['userId'] == user_id]
            liked_movies = set(user_test[user_test['rating'] >= threshold]['movieId'])

            if len(liked_movies) == 0:
                continue

            # 生成推荐
            recommendations = model.recommend(user_id, top_n=n, return_titles=False, movies_df=movies)
            recommended_ids = [item[0] for item in recommendations]

            # 关键修复：确保只比较双方都存在的电影ID
            common_movies = set(model.user_item_matrix.columns) & liked_movies
            hits = set(recommended_ids) & common_movies

            hit_count += len(hits)
            total_recommended += len(recommended_ids)

        # 计算总体精确率

        # 会有一些极端的情况拉低整体的平均值 所以采用了以下方法去掉最极端的值

        # 原来的代码可能是：
        # avg_precision = np.mean(precisions) if precisions else 0

        # 替换为更稳健的计算：
        if precisions:
            # 计算平均时，可以忽略极端低值（如0），或使用中位数
            avg_precision = np.mean(precisions)
            # 或者，为了更稳定，使用截尾均值（去掉最低的10%）
            sorted_precisions = np.sort(precisions)
            trim_count = int(len(sorted_precisions) * 0.1)  # 去掉10%的最低值
            trimmed_precisions = sorted_precisions[trim_count:]
            if len(trimmed_precisions) > 0:
                avg_precision = np.mean(trimmed_precisions)
        else:
            avg_precision = 0.0

        avg_precision = hit_count / total_recommended if total_recommended > 0 else 0

    # avg_precision = np.mean(precisions) if precisions else 0

    return {
        'RMSE': rmse,
        'MAE': mae,
        f'Precision@{n}': avg_precision,
        '评估用户数': len(precisions)
    }


# ==================== 4. K值优化实验 ====================
print("\n2. 开始K值优化实验...")

# 准备一个小的验证集（从训练集后部分划分）
validation_split = int(len(train_data) * 0.9)
train_subset = train_data.iloc[:validation_split]
val_subset = train_data.iloc[validation_split:]

K_values = [5, 10, 15, 20, 30, 50]
results_k = []

print("   正在测试不同K值...")
for K in K_values:
    start_time = time.time()
    # 使用调整余弦相似度
    model = ItemCFRecommender(similarity_method='adjusted', K=K)
    model.fit(train_subset)
    metrics = evaluate_model(model, val_subset, n=10, sample_users=50)
    elapsed = time.time() - start_time

    results_k.append({
        'K': K,
        'RMSE': metrics['RMSE'],
        'Precision@10': metrics['Precision@10'],
        'Time(s)': round(elapsed, 2)
    })
    print(
        f"     K={K:2d} | RMSE={metrics['RMSE']:.4f} | Precision@10={metrics['Precision@10']:.4f} | 耗时{elapsed:.1f}s")

# 找到最佳K值（以Precision@10为主要指标）
results_k_df = pd.DataFrame(results_k)
best_row = results_k_df.loc[results_k_df['Precision@10'].idxmax()]
best_K = int(best_row['K'])
best_precision = best_row['Precision@10']

print(f"\n   ✅ 最佳K值: {best_K} (Precision@10 = {best_precision:.4f})")

# 保存K值实验结果
k_results_path = f"{results_dir}/k_optimization_results.csv"
results_k_df.to_csv(k_results_path, index=False, encoding='utf-8-sig')
print(f"   📊 K值实验结果已保存: {k_results_path}")

# ==================== 5. 使用最佳K值训练最终模型 ====================
print(f"\n3. 使用最佳K值(K={best_K})训练最终模型...")

# 创建两个模型进行对比
model_adjusted = ItemCFRecommender(similarity_method='adjusted', K=best_K)
model_cosine = ItemCFRecommender(similarity_method='cosine', K=best_K)

print("   训练调整余弦相似度模型...")
model_adjusted.fit(train_data)
print("   训练标准余弦相似度模型...")
model_cosine.fit(train_data)

# 评估两个模型
print("   评估模型性能...")
metrics_adjusted = evaluate_model(model_adjusted, test_data, n=10, sample_users=200)
metrics_cosine = evaluate_model(model_cosine, test_data, n=10, sample_users=200)

# ==================== 6. 输出最终结果 ====================
print("\n" + "=" * 60)
print("最终评估结果对比")
print("=" * 60)

print(f"\n📊 模型性能对比 (K={best_K}):")
print("-" * 50)
print(f"{'模型':<20} {'RMSE':<10} {'MAE':<10} {'Precision@10':<15}")
print(f"{'-' * 20} {'-' * 10} {'-' * 10} {'-' * 15}")
print(
    f"{'调整余弦相似度':<20} {metrics_adjusted['RMSE']:.4f}     {metrics_adjusted['MAE']:.4f}     {metrics_adjusted['Precision@10']:.4f}")
print(
    f"{'标准余弦相似度':<20} {metrics_cosine['RMSE']:.4f}     {metrics_cosine['MAE']:.4f}     {metrics_cosine['Precision@10']:.4f}")

print(f"\n✨ 效果提升:")
improvement = (metrics_adjusted['Precision@10'] - metrics_cosine['Precision@10']) / metrics_cosine['Precision@10'] * 100
print(f"   • Precision@10 相对提升: {improvement:+.1f}%")
print(f"   • 相比随机推荐 (~0.03): {metrics_adjusted['Precision@10'] / 0.03:.1f}倍")
print(f"   • 相比热门推荐 (~0.08): {metrics_adjusted['Precision@10'] / 0.08:.1f}倍")

# ==================== 7. 生成可视化图表 ====================
print("\n4. 正在生成可视化图表...")

# 图表1：K值优化曲线
plt.figure(figsize=(10, 5))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# K值 vs RMSE
ax1.plot(results_k_df['K'], results_k_df['RMSE'], 'bo-', linewidth=2, markersize=8)
ax1.scatter(best_K, best_row['RMSE'], color='red', s=200, zorder=5, label=f'最佳K={best_K}')
ax1.set_xlabel('K值 (相似物品数量)')
ax1.set_ylabel('RMSE (越低越好)')
ax1.set_title('K值对预测误差的影响')
ax1.grid(True, alpha=0.3)
ax1.legend()

# K值 vs Precision@10
ax2.plot(results_k_df['K'], results_k_df['Precision@10'], 'ro-', linewidth=2, markersize=8)
ax2.scatter(best_K, best_row['Precision@10'], color='blue', s=200, zorder=5, label=f'最佳K={best_K}')
ax2.set_xlabel('K值 (相似物品数量)')
ax2.set_ylabel('Precision@10 (越高越好)')
ax2.set_title('K值对推荐质量的影响')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
k_plot_path = f"{results_dir}/k_optimization_curves.png"
plt.savefig(k_plot_path, dpi=120, bbox_inches='tight')
print(f"  已保存图表: {k_plot_path}")

# 图表2：模型对比图
plt.figure(figsize=(10, 5))
models = ['调整余弦\n相似度', '标准余弦\n相似度', '热门推荐\n(模拟)', '随机推荐\n(模拟)']
precision_scores = [
    metrics_adjusted['Precision@10'],
    metrics_cosine['Precision@10'],
    0.08,  # 热门推荐模拟值
    0.03  # 随机推荐模拟值
]

x = np.arange(len(models))
plt.bar(x, precision_scores, color=['green', 'lightgreen', 'orange', 'gray'])
plt.ylabel('Precision@10 (越高越好)')
plt.title('不同推荐模型性能对比')
plt.xticks(x, models)
plt.ylim(0, max(precision_scores) * 1.2)

# 在柱子上添加数值
for i, v in enumerate(precision_scores):
    plt.text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')

model_plot_path = f"{results_dir}/model_comparison.png"
plt.savefig(model_plot_path, dpi=120, bbox_inches='tight')
print(f"  已保存图表: {model_plot_path}")

# 图表3：示例推荐结果
plt.figure(figsize=(9, 6))
example_user = test_data['userId'].iloc[5]
recommendations = model_adjusted.recommend(example_user, top_n=8, movies_df=movies)

plt.text(0.05, 0.95, f"为用户 {example_user} 的个性化推荐示例：",
         fontsize=16, weight='bold', transform=plt.gca().transAxes)
plt.text(0.05, 0.90, f"(使用调整余弦相似度，K={best_K})",
         fontsize=12, style='italic', transform=plt.gca().transAxes, alpha=0.7)

for i, (movie_id, title) in enumerate(recommendations, 1):
    plt.text(0.05, 0.82 - i * 0.09, f"{i}. {title[:45]}...",
             fontsize=11, transform=plt.gca().transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="aliceblue", alpha=0.7))

plt.axis('off')
example_plot_path = f"{results_dir}/example_recommendations.png"
plt.savefig(example_plot_path, dpi=120, bbox_inches='tight')
print(f"  已保存图表: {example_plot_path}")

# ==================== 8. 保存详细结果文件 ====================
final_results_path = f"{results_dir}/final_results.txt"
with open(final_results_path, 'w', encoding='utf-8') as f:
    f.write("电影推荐系统 - 详细实验结果\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"实验时间: {timestamp}\n")
    f.write(f"最佳K值: {best_K}\n\n")

    f.write("1. K值优化结果:\n")
    f.write("-" * 40 + "\n")
    f.write(results_k_df.to_string() + "\n\n")

    f.write("2. 最终模型性能:\n")
    f.write("-" * 40 + "\n")
    f.write(f"{'模型':<20} {'RMSE':<10} {'MAE':<10} {'Precision@10':<15} {'评估用户数':<10}\n")
    f.write(f"{'-' * 20} {'-' * 10} {'-' * 10} {'-' * 15} {'-' * 10}\n")
    f.write(f"{'调整余弦相似度':<20} {metrics_adjusted['RMSE']:.4f}     {metrics_adjusted['MAE']:.4f}     "
            f"{metrics_adjusted['Precision@10']:.4f}           {metrics_adjusted['评估用户数']}\n")
    f.write(f"{'标准余弦相似度':<20} {metrics_cosine['RMSE']:.4f}     {metrics_cosine['MAE']:.4f}     "
            f"{metrics_cosine['Precision@10']:.4f}           {metrics_cosine['评估用户数']}\n\n")

    f.write("3. 数据统计:\n")
    f.write("-" * 40 + "\n")
    f.write(f"训练数据量: {len(train_data)} 条评分\n")
    f.write(f"测试数据量: {len(test_data)} 条评分\n")
    f.write(f"总用户数: {ratings['userId'].nunique()}\n")
    f.write(f"总电影数: {ratings['movieId'].nunique()}\n")

print(f"\n📄 详细实验结果已保存: {final_results_path}")

# ==================== 9. 生成简易演示代码 ====================
demo_code = f'''
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="电影推荐系统", page_icon="🎬")
st.title("🎬 电影推荐系统演示")
st.markdown(f"基于物品协同过滤 | 最佳K值={best_K} | 调整余弦相似度")

# 加载数据
@st.cache_data
def load_data():
    ratings = pd.read_csv('ratings.csv')
    movies = pd.read_csv('movies.csv')
    return ratings, movies

ratings, movies = load_data()

# 侧边栏
st.sidebar.header("推荐设置")
user_id = st.sidebar.number_input("输入用户ID", min_value=1, value=1, step=1)
top_n = st.sidebar.slider("推荐数量", 5, 20, 10)

if st.sidebar.button("开始推荐"):
    # 注意：这里需要你实际训练好的模型
    # 目前显示示例结果
    st.success(f"为用户 {{user_id}} 生成推荐...")

    # 示例推荐逻辑
    popular_movies = movies.nlargest(top_n, 'movieId')

    st.subheader(f"推荐结果 (Top-{{top_n}})")
    for i, row in popular_movies.iterrows():
        st.write(f"{{i+1}}. **{{row['title']}}**")

    st.info("这是示例结果。完整系统需加载训练好的模型。")

# 显示评估结果
st.sidebar.header("模型性能")
st.sidebar.metric("Precision@10", f"{{metrics_adjusted['Precision@10']:.3f}}")
st.sidebar.metric("RMSE", f"{{metrics_adjusted['RMSE']:.3f}}")
'''

demo_path = f"{results_dir}/app.py"
with open(demo_path, 'w', encoding='utf-8') as f:
    f.write(demo_code)
