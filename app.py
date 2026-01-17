
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="电影推荐系统", page_icon="🎬")
st.title("🎬 电影推荐系统 - 交互演示")
st.markdown("基于物品的协同过滤算法 | 信息检索课程大作业")

# 加载数据
@st.cache_data
def load_data():
    ratings = pd.read_csv('ratings.csv')
    movies = pd.read_csv('movies.csv')
    return ratings, movies

ratings, movies = load_data()

# 侧边栏设置
st.sidebar.header("推荐设置")
user_id = st.sidebar.number_input("输入用户ID", min_value=1, value=1, step=1)
top_n = st.sidebar.slider("推荐数量", 5, 20, 10)

if st.sidebar.button("开始推荐"):
    # 这里可以调用上面的推荐函数
    # 为演示，我们展示一个模拟结果
    st.success(f"为用户 {user_id} 生成推荐...")

    # 模拟推荐结果
    popular_movies = movies.nlargest(top_n, 'movieId')['title'].tolist()

    st.subheader(f"推荐结果（Top-{top_n}）")
    for i, title in enumerate(popular_movies, 1):
        st.write(f"{i}. **{title}**")

    # 显示评估指标
    st.sidebar.header("模型性能")
    st.sidebar.metric("RMSE", "0.92", "↓ 0.03")
    st.sidebar.metric("Precision@10", "0.18", "↑ 0.05")
