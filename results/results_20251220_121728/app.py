
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="电影推荐系统", page_icon="🎬")
st.title("🎬 电影推荐系统演示")
st.markdown(f"基于物品协同过滤 | 最佳K值=5 | 调整余弦相似度")

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
    st.success(f"为用户 {user_id} 生成推荐...")

    # 示例推荐逻辑
    popular_movies = movies.nlargest(top_n, 'movieId')

    st.subheader(f"推荐结果 (Top-{top_n})")
    for i, row in popular_movies.iterrows():
        st.write(f"{i+1}. **{row['title']}**")

    st.info("这是示例结果。完整系统需加载训练好的模型。")

# 显示评估结果
st.sidebar.header("模型性能")
st.sidebar.metric("Precision@10", f"{metrics_adjusted['Precision@10']:.3f}")
st.sidebar.metric("RMSE", f"{metrics_adjusted['RMSE']:.3f}")
