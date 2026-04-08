from pathlib import Path
import json

import streamlit as st
import xgboost as xgb
from utils import load_embedding_payload, load_hero_id_to_name, load_hero_id_to_url_name
from inference import XGBInference, TransformerInference
import plotly.express as px

st.set_page_config(page_title="LatentDraft - Dota 2 阵容助手", layout="wide")

st.markdown(
    """
    <style>
    div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
        min-height: 1.55rem;
        padding: 0.08rem 0.35rem;
        font-size: 0.72rem;
        line-height: 1.0;
        border-radius: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_engine(model_type, model_path, embedding_path, hero_name_path):
    # 通用数据加载
    hero_id_to_name = load_hero_id_to_name(Path(hero_name_path))
    num_heroes = max(hero_id_to_name.keys())
    if model_type == "XGBoost":
        hero_pool, emb_tensor = load_embedding_payload(Path(embedding_path))
        engine = XGBInference(model_path, emb_tensor.cpu().numpy())
        valid_ids = sorted(set(hero_pool) & set(hero_id_to_name.keys()))
    else:
        # Transformer 默认从模型里推断，不依赖外部 Embedding 文件
        engine = TransformerInference(model_path, num_heroes=num_heroes)
        valid_ids = sorted(hero_id_to_name.keys())
        
    return engine, hero_id_to_name, valid_ids


def fallback_url_name(hero_name: str) -> str:
    s = hero_name.lower().strip()
    s = s.replace("'", "")
    s = s.replace("-", "_")
    s = s.replace(" ", "_")
    s = s.replace(".", "")
    return s


if "ally_team" not in st.session_state:
    st.session_state.ally_team = []
if "enemy_team" not in st.session_state:
    st.session_state.enemy_team = []


with st.expander("⚙️ 模型配置与设置 (选好后可折叠以释放空间)", expanded=False):
    # 横向排列三个设置项，极限压缩垂直空间
    cfg_cols = st.columns(3)
    
    with cfg_cols[0]:
        m_type = st.selectbox("核心算法", ["Transformer", "XGBoost"])
    
    with cfg_cols[1]:
        if m_type == "XGBoost":
            m_path = st.text_input("XGB File", "xgb_bp.model")
        else:
            m_path = st.text_input("Transformer File", "dota_bert.pt")
            
    with cfg_cols[2]:
        if m_type == "XGBoost":
            e_path = st.text_input("Emb File", "hero_embedding.pt")
        else:
            e_path = None
            st.caption("Transformer 无需外部 Emb")
hero_name_path = "hero_id_to_name.json"
# 加载引擎
engine, hero_id_to_name, valid_hero_ids = load_engine(m_type, m_path, e_path, hero_name_path)


def add_hero(hero_id: int, side: str) -> None:
    if hero_id in st.session_state.ally_team or hero_id in st.session_state.enemy_team:
        return
    if side == "己方":
        if len(st.session_state.ally_team) < 5:
            st.session_state.ally_team.append(hero_id)
    else:
        if len(st.session_state.enemy_team) < 5:
            st.session_state.enemy_team.append(hero_id)


def remove_hero(hero_id: int, side: str) -> None:
    if side == "己方":
        st.session_state.ally_team = [h for h in st.session_state.ally_team if h != hero_id]
    else:
        st.session_state.enemy_team = [h for h in st.session_state.enemy_team if h != hero_id]
hero_name_path = "hero_id_to_url_name.json"
hero_id_to_url_name = load_hero_id_to_url_name(Path(hero_name_path))
def hero_image_url(hero_id: int, hero_name: str) -> str:
    url_name = hero_id_to_url_name.get(hero_id, fallback_url_name(hero_name))
    return f"https://cdn.cloudflare.steamstatic.com/apps/dota2/images/dota_react/heroes/{url_name}.png"


def render_selected_team(title: str, team: list, side: str, key_prefix: str) -> None:
    st.write(f"{title}:")
    if not team:
        st.caption("(空)")
        return

    cols = st.columns(5)
    for i, hero_id in enumerate(team):
        hero_name = hero_id_to_name.get(hero_id, "Unknown")
        with cols[i % 5]:
            st.image(hero_image_url(hero_id, hero_name), width=70, use_container_width=False)
            if st.button("删除", key=f"{key_prefix}_{hero_id}_{i}"):
                remove_hero(hero_id, side)
                st.rerun()


def team_text(team):
    if not team:
        return "(空)"
    return ", ".join([f"{hid}:{hero_id_to_name.get(hid, 'Unknown')}" for hid in team])


left_col, right_col = st.columns([2, 1])



with left_col:
    winrate_placeholder = st.empty()  # 用于后续动态更新胜率显示

    st.subheader("已选阵容")
    render_selected_team("己方", st.session_state.ally_team, "己方", "rm_ally")
    render_selected_team("敌方", st.session_state.enemy_team, "敌方", "rm_enemy")

    cols = st.columns(10)
    for idx, hero_id in enumerate(valid_hero_ids):
        hero_name = hero_id_to_name.get(hero_id, "Unknown")
        disabled = hero_id in st.session_state.ally_team or hero_id in st.session_state.enemy_team
        ally_full = len(st.session_state.ally_team) >= 5
        enemy_full = len(st.session_state.enemy_team) >= 5
        with cols[idx % 10]:
            st.image(hero_image_url(hero_id, hero_name))
            btn_cols = st.columns([1, 1])
            with btn_cols[0]:
                if st.button("己", key=f"ally_{hero_id}", disabled=disabled or ally_full):
                    add_hero(hero_id, "己方")
                    st.rerun()
            with btn_cols[1]:
                if st.button("敌", key=f"enemy_{hero_id}", disabled=disabled or enemy_full):
                    add_hero(hero_id, "敌方")
                    st.rerun()

with right_col:
    st.title("实时建议")
    if not st.session_state.ally_team and not st.session_state.enemy_team:
        st.info("先在左侧选择英雄，再查看推荐")
    else:
        pick_results = engine.recommend(
            current_ally=st.session_state.ally_team,
            current_enemy=st.session_state.enemy_team,
            valid_hero_ids=valid_hero_ids,
            mode="pick"
        )
        ban_results = engine.recommend(
            current_ally=st.session_state.ally_team,
            current_enemy=st.session_state.enemy_team,
            valid_hero_ids=valid_hero_ids,
            mode="ban"
        )

        st.subheader("推荐 Pick")
        for i, (hero_id, score) in enumerate(pick_results):
            name = hero_id_to_name.get(int(hero_id), "Unknown")
            
            # 使用 container 让每个英雄的展示更紧凑
            with st.container():
                cols = st.columns([1, 4])
                with cols[0]:
                    # 显示英雄小头像
                    st.image(hero_image_url(hero_id, name), use_container_width=True)
                with cols[1]:
                    st.markdown(f"**{i+1}. {name}** (预测得分: `{float(score):.2f}`)")
                    if i < 5 and (st.session_state.ally_team or st.session_state.enemy_team):
                        enemy_bonds, ally_bonds = engine.get_explanation(
                            hero_id, 
                            st.session_state.ally_team, 
                            st.session_state.enemy_team
                        )
                        reasons = []
                        for e_id, e_val, e_delta in enemy_bonds:
                            e_name = hero_id_to_name.get(e_id, "Unknown")
                            reasons.append(f"克制: {e_name} {e_val:+.2f} {e_delta:+.2f}")
                        for a_id, a_val, a_delta in ally_bonds:
                            a_name = hero_id_to_name.get(a_id, "Unknown")
                            reasons.append(f"配合: {a_name} {a_val:+.2f} {a_delta:+.2f}")
                        if reasons:
                            st.caption(" | ".join(reasons))

        st.divider()
        st.subheader("推荐 Ban")
        for j, (hero_id, score) in enumerate(ban_results):
            name = hero_id_to_name.get(int(hero_id), "Unknown")
            with st.container():
                cols = st.columns([1, 4])
                with cols[0]:
                    # 显示英雄小头像
                    st.image(hero_image_url(hero_id, name), use_container_width=True)
                with cols[1]:
                    st.markdown(f"**{j+1}. {name}** (预测得分: `{float(score):.2f}`)")

st.divider()
with st.expander("🔍 一键全显：10x10 全景战术矩阵", expanded=True):
    if m_type == "Transformer":
        try:
            # 1. 准备 10 人阵容数据（不足 5 人用 ID 0 补齐）
            full_ally = (st.session_state.ally_team + [0]*5)[:5]
            full_enemy = (st.session_state.enemy_team + [0]*5)[:5]
            current_heroes = full_ally + full_enemy
            current_sides = [0]*5 + [1]*5
            
            # 2. 调用后端接口获取矩阵和抖动数据
            # 确保你在 inference.py 的 TransformerInference 类里实现了 get_full_analysis
            matrix, deltas, base_prob = engine.get_full_analysis(current_heroes, current_sides)
            
            color = "green" if base_prob >= 0.5 else "red"
            winrate_placeholder.markdown(
                f"<div style='text-align: right; color: {color}; font-size: 22px; font-weight: bold; margin-top: 25px;'>当前胜率: {base_prob * 100:.1f}%</div>", 
                unsafe_allow_html=True
            )

            # 3. 构造坐标轴标签：英雄名 + 抖动权重
            hero_labels = []
            for i, h_id in enumerate(current_heroes):
                name = hero_id_to_name.get(h_id, "空位")
                d_val = deltas[i]
                hero_labels.append(f"{name}<br>({d_val:+.2f})")

            # 4. 绘制交互式热力图
            fig = px.imshow(
                matrix,
                x=hero_labels,
                y=hero_labels,
                labels=dict(x="被关注目标 (Target)", y="关注发起者 (Source)", color="注意力 (%)"),
                color_continuous_scale="RdBu_r", # 红蓝配色，深色代表强关注
                text_auto=".1f",
                aspect="auto"
            )

            # 5. 美化布局：增加十字分割线区分敌我
            fig.add_shape(type="line", x0=4.5, y0=-0.5, x1=4.5, y1=9.5, line=dict(color="white", width=3))
            fig.add_shape(type="line", x0=-0.5, y0=4.5, x1=9.5, y1=4.5, line=dict(color="white", width=3))
            
            fig.update_layout(
                title="全场英雄互视矩阵 (越红代表针对性越强)",
                xaxis_title="防御/配合侧 (谁被盯着？)",
                yaxis_title="进攻/控制侧 (谁在盯着？)",
                height=700
            )

            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"winrate: {base_prob:.4f} 💡 **如何阅读**：右上角区域是‘我方对敌方的针对’，左下角区域是‘敌方对我方的威胁’。括号内的数字是抖动 Delta，负值越大代表该英雄对全局胜率的负面压力越大。")

        except Exception as e:
            st.error(f"矩阵渲染失败，请检查后端 get_full_analysis 函数: {e}")
    else:
        st.warning("XGBoost 不支持注意力矩阵显示。")