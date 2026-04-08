from pathlib import Path
import json

import streamlit as st
import xgboost as xgb

from recommend import load_embedding_payload, load_hero_id_to_name, recommend, get_ablation_explanation


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
def load_runtime(model_path: str, embedding_path: str, hero_name_path: str, hero_url_path: str):
    hero_pool, emb_tensor = load_embedding_payload(Path(embedding_path))
    hero_embeddings = emb_tensor.cpu().numpy()

    bst_model = xgb.Booster()
    bst_model.load_model(model_path)

    hero_id_to_name = load_hero_id_to_name(Path(hero_name_path))
    with Path(hero_url_path).open("r", encoding="utf-8") as f:
        raw_url = json.load(f)
    hero_id_to_url_name = {int(k): str(v) for k, v in raw_url.items()}

    valid_hero_ids = sorted(set(hero_pool) & set(hero_id_to_name.keys()))
    return bst_model, hero_embeddings, hero_id_to_name, hero_id_to_url_name, valid_hero_ids


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


with st.sidebar:
    st.header("模型配置")
    model_path = st.text_input("XGB Model", value="xgb_bp.model")
    embedding_path = st.text_input("Embedding", value="hero_embedding.pt")
    hero_name_path = st.text_input("Hero Name JSON", value="hero_id_to_name.json")
    hero_url_path = st.text_input("Hero URL JSON", value="hero_id_to_url_name.json")

    st.header("阵容控制")
    if st.button("重置所有选人"):
        st.session_state.ally_team = []
        st.session_state.enemy_team = []
        st.rerun()

    target_side = st.radio("点击英雄后加入", ["己方", "敌方"], horizontal=True)


try:
    bst_model, hero_embeddings, hero_id_to_name, hero_id_to_url_name, valid_hero_ids = load_runtime(
        model_path=model_path,
        embedding_path=embedding_path,
        hero_name_path=hero_name_path,
        hero_url_path=hero_url_path,
    )
except Exception as e:
    st.error(f"加载模型或数据失败: {e}")
    st.stop()


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
    st.title("英雄选择")
    st.caption("每个英雄图下面有两个半按钮：左己、右敌")

    st.subheader("已选阵容")
    render_selected_team("己方", st.session_state.ally_team, "己方", "rm_ally")
    render_selected_team("敌方", st.session_state.enemy_team, "敌方", "rm_enemy")

    cols = st.columns(8)
    for idx, hero_id in enumerate(valid_hero_ids):
        hero_name = hero_id_to_name.get(hero_id, "Unknown")
        disabled = hero_id in st.session_state.ally_team or hero_id in st.session_state.enemy_team
        ally_full = len(st.session_state.ally_team) >= 5
        enemy_full = len(st.session_state.enemy_team) >= 5
        with cols[idx % 8]:
            st.image(hero_image_url(hero_id, hero_name), width=62, use_container_width=False)
            btn_cols = st.columns([1, 1], gap="small")
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
        pick_results = recommend(
            current_ally=st.session_state.ally_team,
            current_enemy=st.session_state.enemy_team,
            bst_model=bst_model,
            hero_embeddings=hero_embeddings,
            valid_hero_ids=valid_hero_ids,
            mode="pick",
            topk=10,
        )
        ban_results = recommend(
            current_ally=st.session_state.ally_team,
            current_enemy=st.session_state.enemy_team,
            bst_model=bst_model,
            hero_embeddings=hero_embeddings,
            valid_hero_ids=valid_hero_ids,
            mode="ban",
            topk=10,
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
                    
                    # 💡 核心：为 Top 5 英雄提供 AI 解释引擎
                    if i < 5 and (st.session_state.ally_team or st.session_state.enemy_team):
                        enemy_deltas, ally_deltas = get_ablation_explanation(
                            hero_id, st.session_state.ally_team, st.session_state.enemy_team, 
                            bst_model, hero_embeddings
                        )
                        
                        reasons = []
                        # 阈值设定：如果贡献度大于 1%，我们就认为它是一个有效的战术针对
                        for counter_id, counter_delta in enemy_deltas:
                            c_name = hero_id_to_name.get(counter_id, "Unknown")
                            reasons.append(f"⚔️ **克制**: {c_name} ({counter_delta*100:.1f}%)")
                            
                        for ally_id, ally_delta in ally_deltas:
                            a_name = hero_id_to_name.get(ally_id, "Unknown")
                            reasons.append(f"🤝 **配合**: {a_name} ({ally_delta*100:.1f}%)")
                            
                        if reasons:
                            st.caption(" | ".join(reasons))

        st.divider()
        st.subheader("推荐 Ban")
        for hero_id, score in ban_results:
            name = hero_id_to_name.get(int(hero_id), "Unknown")
            st.write(f"{hero_id}: {name}  |  score={float(score):.4f}")