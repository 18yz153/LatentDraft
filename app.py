from pathlib import Path
import json
import numpy as np
import streamlit as st
import xgboost as xgb
from src.utils import load_embedding_payload, load_hero_id_to_name, load_hero_id_to_url_name
from inference import XGBInference, TransformerInference
import plotly.express as px

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models" / "final"

TRANSLATIONS = {
    "zh": {
        "page_title": "LatentDraft - Dota 2 阵容助手",
        "language": "Language / 语言",
        "config_expander": "⚙️ 设置 (选好后可折叠以释放空间)",
        "algo": "核心算法",
        "xgb_file": "XGB File",
        "transformer_file": "Transformer File",
        "emb_file": "Emb File",
        "transformer_no_emb": "Transformer 无需外部 Emb",
        "selected_lineup": "已选阵容",
        "ally": "己方",
        "enemy": "敌方",
        "empty": "(空)",
        "delete": "删除",
        "hero_filter": "筛选英雄（英文）",
        "hero_filter_placeholder": "例如：p、spirit，axe",
        "match_mode": "匹配方式",
        "prefix": "前缀",
        "contains": "包含",
        "filter_result": "筛选结果: {count} / {total}",
        "pick_short": "己",
        "ban_short": "敌",
        "realtime_reco": "实时建议",
        "select_hero_hint": "先在左侧选择英雄，再查看推荐",
        "reco_pick": "推荐 Pick",
        "reco_ban": "推荐 Ban",
        "pred_score": "预测得分",
        "counter": "克制",
        "synergy": "配合",
        "matrix_expander": "🔍 一键全显：10x10 全景战术矩阵",
        "current_winrate": "当前胜率: {winrate:.1f}%",
        "empty_slot": "空位",
        "heatmap_target": "被关注目标 (Target)",
        "heatmap_source": "关注发起者 (Source)",
        "heatmap_attn": "注意力 (%)",
        "heatmap_title": "全场英雄互视矩阵 (越红代表针对性越强)",
        "heatmap_x": "防御/配合侧 (谁被盯着？)",
        "heatmap_y": "进攻/控制侧 (谁在盯着？)",
        "matrix_read": "winrate: {prob:.4f} 右上角区域是我方对敌方的针对，左下角区域是敌方对我方的威胁。括号内数字是抖动 Delta，负值越大代表该英雄对全局胜率的负面压力越大。",
        "matrix_fail": "矩阵渲染失败，请检查后端 get_full_analysis 函数: {err}",
        "xgb_no_matrix": "XGBoost 不支持注意力矩阵显示。",
        "unknown": "Unknown",
    },
    "en": {
        "page_title": "LatentDraft - Dota 2 Draft Assistant",
        "language": "Language / 语言",
        "config_expander": "⚙️ Settings (collapse after setup)",
        "algo": "Core Algorithm",
        "xgb_file": "XGB File",
        "transformer_file": "Transformer File",
        "emb_file": "Embedding File",
        "transformer_no_emb": "Transformer does not require external embedding",
        "selected_lineup": "Selected Lineup",
        "ally": "Ally",
        "enemy": "Enemy",
        "empty": "(Empty)",
        "delete": "Remove",
        "hero_filter": "Filter heroes",
        "hero_filter_placeholder": "e.g. p, spirit, axe",
        "match_mode": "Match Mode",
        "prefix": "Prefix",
        "contains": "Contains",
        "filter_result": "Filter result: {count} / {total}",
        "pick_short": "Ally",
        "ban_short": "Enemy",
        "realtime_reco": "Live Suggestions",
        "select_hero_hint": "Pick heroes on the left first, then view recommendations.",
        "reco_pick": "Recommended Picks",
        "reco_ban": "Recommended Bans",
        "pred_score": "Predicted Score",
        "counter": "Counter",
        "synergy": "Synergy",
        "matrix_expander": "🔍 Full 10x10 Tactical Matrix",
        "current_winrate": "Current Win Rate: {winrate:.1f}%",
        "empty_slot": "Empty",
        "heatmap_target": "Target",
        "heatmap_source": "Source",
        "heatmap_attn": "Attention (%)",
        "heatmap_title": "Hero Interaction Matrix (red means stronger pressure)",
        "heatmap_x": "Defense/Synergy Side (who is targeted)",
        "heatmap_y": "Attack/Control Side (who is targeting)",
        "matrix_read": "winrate: {prob:.4f} The top-right quadrant shows your pressure against enemies; bottom-left shows enemy pressure against your team. Delta in parentheses indicates sensitivity when a hero is removed.",
        "matrix_fail": "Matrix rendering failed. Please check get_full_analysis: {err}",
        "xgb_no_matrix": "XGBoost does not support attention matrix visualization.",
        "unknown": "Unknown",
    },
}


def t(key: str, **kwargs) -> str:
    text = TRANSLATIONS[LANG].get(key, key)
    if kwargs:
        return text.format(**kwargs)
    return text


def list_model_files(suffixes=None):
    if not MODELS_DIR.exists():
        st.warning(f"Model folder not found: {MODELS_DIR}")
        return []

    files = [p.name for p in MODELS_DIR.iterdir() if p.is_file()]
    if suffixes is not None:
        files = [name for name in files if Path(name).suffix.lower() in suffixes]
    return sorted(files)

if "ui_lang" not in st.session_state:
    st.session_state.ui_lang = "中文"

LANG = "zh" if st.session_state.ui_lang == "中文" else "en"

st.set_page_config(page_title=t("page_title"), layout="wide")

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
    hero_id_to_name = load_hero_id_to_name()
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
if "explanation_cache" not in st.session_state:
    st.session_state.explanation_cache = {}


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
hero_name_path = str(DATA_DIR / "hero_id_to_url_name.json")
hero_id_to_url_name = load_hero_id_to_url_name(Path(hero_name_path))
def hero_image_url(hero_id: int, hero_name: str) -> str:
    url_name = hero_id_to_url_name.get(hero_id, fallback_url_name(hero_name))
    return f"https://cdn.cloudflare.steamstatic.com/apps/dota2/images/dota_react/heroes/{url_name}.png"


def render_selected_team(title: str, team: list, side: str, key_prefix: str) -> None:
    st.write(f"{title}:")
    if not team:
        st.caption(t("empty"))
        return

    cols = st.columns(5)
    for i, hero_id in enumerate(team):
        hero_name = hero_id_to_name.get(hero_id, "Unknown")
        with cols[i % 5]:
            st.image(hero_image_url(hero_id, hero_name), width=70, use_container_width=False)
            if st.button(t("delete"), key=f"{key_prefix}_{hero_id}_{i}"):
                remove_hero(hero_id, side)
                st.rerun()


def team_text(team):
    if not team:
        return "(空)"
    return ", ".join([f"{hid}:{hero_id_to_name.get(hid, 'Unknown')}" for hid in team])


def match_hero_name(hero_name: str, keyword: str, mode: str) -> bool:
    if not keyword:
        return True
    name_l = hero_name.lower()
    key_l = keyword.lower()
    if mode == "prefix":
        return name_l.startswith(key_l)
    return key_l in name_l


def build_inference_state(engine, m_type, m_path, ally_team, enemy_team, valid_hero_ids):
    state = {
        "pick_results": [],
        "ban_results": [],
        "lineup_cache_key": (m_type, m_path, tuple(ally_team), tuple(enemy_team)),
        "pick_explanations": {},
        "matrix": None,
        "deltas": None,
        "base_prob": None,
        "matrix_error": None,
    }

    if not ally_team and not enemy_team:
        return state

    if len(ally_team) < 5:
        state["pick_results"] = engine.recommend(
            current_ally=ally_team,
            current_enemy=enemy_team,
            valid_hero_ids=valid_hero_ids,
            mode="pick",
        )

    if len(enemy_team) < 5:
        state["ban_results"] = engine.recommend(
            current_ally=ally_team,
            current_enemy=enemy_team,
            valid_hero_ids=valid_hero_ids,
            mode="ban",
        )

    if state["pick_results"] and (ally_team or enemy_team):
        for i, (hero_id, _) in enumerate(state["pick_results"][:5]):
            explain_key = state["lineup_cache_key"] + (int(hero_id),)
            if explain_key in st.session_state.explanation_cache:
                enemy_bonds, ally_bonds = st.session_state.explanation_cache[explain_key]
            else:
                enemy_bonds, ally_bonds = engine.get_explanation(
                    hero_id,
                    ally_team,
                    enemy_team,
                )
                st.session_state.explanation_cache[explain_key] = (enemy_bonds, ally_bonds)
            state["pick_explanations"][int(hero_id)] = (enemy_bonds, ally_bonds)

    if m_type == "Transformer":
        try:
            full_ally = (ally_team + [0] * 5)[:5]
            full_enemy = (enemy_team + [0] * 5)[:5]
            current_heroes = full_ally + full_enemy
            current_sides = [0] * 5 + [1] * 5
            matrix, deltas, base_prob, role_prob = engine.get_full_analysis(current_heroes, current_sides)
            state["matrix"] = matrix
            state["deltas"] = deltas
            state["base_prob"] = base_prob
            state["role_prob"] = role_prob
        except Exception as e:
            state["matrix_error"] = e

    return state


left_col, right_col = st.columns([2, 1])



with left_col:
    top_row = st.columns([3.2, 1.0])
    with top_row[0]:
        with st.expander(t("config_expander"), expanded=False):
            cfg_cols = st.columns([1.1, 1.0, 1.6, 1.6])
            xgb_files = list_model_files({".model", ".json", ".ubj", ".bin"})
            torch_files = list_model_files({".pt", ".pth"})

            with cfg_cols[0]:
                st.session_state.ui_lang = st.selectbox(
                    t("language"),
                    ["中文", "English"],
                    index=0 if st.session_state.ui_lang == "中文" else 1,
                    label_visibility="collapsed",
                )
                st.caption(t("language"))
                LANG = "zh" if st.session_state.ui_lang == "中文" else "en"
            
            mode = st.secrets["mode"]
            if mode == "dev":
                with cfg_cols[1]:
                    m_type = st.selectbox(t("algo"), ["Transformer", "XGBoost"], label_visibility="collapsed")
                    st.caption(t("algo"))

                with cfg_cols[2]:
                    if m_type == "XGBoost":
                        xgb_options = xgb_files if xgb_files else ["xgb_bp.model"]
                        xgb_default = xgb_options.index("xgb_bp.model") if "xgb_bp.model" in xgb_options else 0
                        selected_xgb = st.selectbox(t("xgb_file"), xgb_options, index=xgb_default, label_visibility="collapsed")
                        m_path = str(MODELS_DIR / selected_xgb)
                        st.caption(t("xgb_file"))
                    else:
                        transformer_options = torch_files if torch_files else ["stage3_value_network_best.pt"]
                        transformer_default = transformer_options.index("stage3_value_network_best.pt") if "stage3_value_network_best.pt" in transformer_options else 0
                        selected_transformer = st.selectbox(t("transformer_file"), transformer_options, index=transformer_default, label_visibility="collapsed")
                        m_path = str(MODELS_DIR / selected_transformer)
                        st.caption(t("transformer_file"))

                with cfg_cols[3]:
                    if m_type == "XGBoost":
                        emb_options = torch_files if torch_files else ["hero_embedding.pt"]
                        emb_default = emb_options.index("hero_embedding.pt") if "hero_embedding.pt" in emb_options else 0
                        selected_emb = st.selectbox(t("emb_file"), emb_options, index=emb_default, label_visibility="collapsed")
                        e_path = str(MODELS_DIR / selected_emb)
                        st.caption(t("emb_file"))
                    else:
                        e_path = None
                        st.caption(t("transformer_no_emb"))
            else:
                m_type = "Transformer"
                m_path = str(MODELS_DIR / "stage3_value_network_best.pt")
                e_path = None

            hero_name_path = str(DATA_DIR / "hero_id_to_name.json")
            engine, hero_id_to_name, valid_hero_ids = load_engine(m_type, m_path, e_path, hero_name_path)

            inference_state = build_inference_state(
                engine=engine,
                m_type=m_type,
                m_path=m_path,
                ally_team=st.session_state.ally_team,
                enemy_team=st.session_state.enemy_team,
                valid_hero_ids=valid_hero_ids,
            )

    with top_row[1]:
        winrate_placeholder = st.empty()  # 用于动态更新胜率显示

    st.subheader(t("selected_lineup"))
    render_selected_team(t("ally"), st.session_state.ally_team, "己方", "rm_ally")
    render_selected_team(t("enemy"), st.session_state.enemy_team, "敌方", "rm_enemy")

    filter_cols = st.columns([2, 1])
    with filter_cols[0]:
        hero_filter_keyword = st.text_input(t("hero_filter"), value="", placeholder=t("hero_filter_placeholder"))
    with filter_cols[1]:
        mode_options = {"prefix": t("prefix"), "contains": t("contains")}
        hero_filter_mode = st.selectbox(t("match_mode"), ["prefix", "contains"], index=0, format_func=lambda x: mode_options[x])

    filtered_hero_ids = [
        hid for hid in valid_hero_ids
        if match_hero_name(hero_id_to_name.get(hid, "Unknown"), hero_filter_keyword.strip(), hero_filter_mode)
    ]

    if hero_filter_keyword.strip():
        st.caption(t("filter_result", count=len(filtered_hero_ids), total=len(valid_hero_ids)))

    cols = st.columns(10)
    for idx, hero_id in enumerate(filtered_hero_ids):
        hero_name = hero_id_to_name.get(hero_id, "Unknown")
        disabled = hero_id in st.session_state.ally_team or hero_id in st.session_state.enemy_team
        ally_full = len(st.session_state.ally_team) >= 5
        enemy_full = len(st.session_state.enemy_team) >= 5
        with cols[idx % 10]:
            st.image(hero_image_url(hero_id, hero_name))
            btn_cols = st.columns([1, 1])
            with btn_cols[0]:
                if st.button(t("pick_short"), key=f"ally_{hero_id}", disabled=disabled or ally_full):
                    add_hero(hero_id, "己方")
                    st.rerun()
            with btn_cols[1]:
                if st.button(t("ban_short"), key=f"enemy_{hero_id}", disabled=disabled or enemy_full):
                    add_hero(hero_id, "敌方")
                    st.rerun()

with right_col:
    st.title(t("realtime_reco"))
    if not st.session_state.ally_team and not st.session_state.enemy_team:
        st.info(t("select_hero_hint"))
    else:
        pick_results = inference_state["pick_results"]
        ban_results = inference_state["ban_results"]

        st.subheader(t("reco_pick"))
        for i, (hero_id, score) in enumerate(pick_results):
            name = hero_id_to_name.get(int(hero_id), t("unknown"))
            
            # 使用 container 让每个英雄的展示更紧凑
            with st.container():
                cols = st.columns([1, 4])
                with cols[0]:
                    # 显示英雄小头像
                    st.image(hero_image_url(hero_id, name), use_container_width=True)
                with cols[1]:
                    st.markdown(f"**{i+1}. {name}** ({t('pred_score')}: `{float(score):.2f}`)")
                    if i < 5 and (st.session_state.ally_team or st.session_state.enemy_team):
                        enemy_bonds, ally_bonds = inference_state["pick_explanations"].get(int(hero_id), ([], []))
                        reasons = []
                        for e_id, e_val, e_delta in enemy_bonds:
                            e_name = hero_id_to_name.get(e_id, t("unknown"))
                            reasons.append(f"{t('counter')}: {e_name} {e_val:+.2f} {e_delta:+.2f}")
                        for a_id, a_val, a_delta in ally_bonds:
                            a_name = hero_id_to_name.get(a_id, t("unknown"))
                            reasons.append(f"{t('synergy')}: {a_name} {a_val:+.2f} {a_delta:+.2f}")
                        if reasons:
                            st.caption(" | ".join(reasons))

        st.divider()
        st.subheader(t("reco_ban"))
        for j, (hero_id, score) in enumerate(ban_results):
            name = hero_id_to_name.get(int(hero_id), t("unknown"))
            with st.container():
                cols = st.columns([1, 4])
                with cols[0]:
                    # 显示英雄小头像
                    st.image(hero_image_url(hero_id, name), use_container_width=True)
                with cols[1]:
                    st.markdown(f"**{j+1}. {name}** ({t('pred_score')}: `{float(score):.2f}`)")

st.divider()
with st.expander(t("matrix_expander"), expanded=True):
    if m_type == "Transformer":
        try:
            matrix = inference_state["matrix"]
            deltas = inference_state["deltas"]
            base_prob = inference_state["base_prob"]
            role_probs = inference_state.get("role_prob")
            if matrix is None or deltas is None or base_prob is None:
                raise RuntimeError(inference_state["matrix_error"] or "matrix unavailable")

            full_ally = (st.session_state.ally_team + [0]*5)[:5]
            full_enemy = (st.session_state.enemy_team + [0]*5)[:5]
            current_heroes = full_ally + full_enemy

            color = "green" if base_prob >= 0.5 else "red"
            winrate_placeholder.markdown(
                f"<div style='text-align: right; color: {color}; font-size: 22px; font-weight: bold; margin-top: 25px;'>{t('current_winrate', winrate=base_prob * 100)}</div>", 
                unsafe_allow_html=True
            )

            # 3. 构造坐标轴标签：英雄名 + 抖动权重
            hero_labels = []
            for i, h_id in enumerate(current_heroes):
                name = hero_id_to_name.get(h_id, t("empty_slot"))
                d_val = deltas[i]
                
                role_info_list = []
                if h_id != 0 and role_probs is not None:
                    probs = role_probs[i] # 获取该英雄 5 个位置的概率
                    
                    # 🌟 核心：找出所有概率显著的位置 (比如 > 20%)
                    # 这样如果一个英雄 40% 打 1，35% 打 3，两个都会显示出来
                    significant_indices = np.where(probs > 0.20)[0] 
                    
                    # 如果没超过 20% 的，就保底拿一个最高的
                    if len(significant_indices) == 0:
                        significant_indices = [np.argmax(probs)]
                    
                    # 按概率从大到小排序显示
                    significant_indices = sorted(significant_indices, key=lambda idx: probs[idx], reverse=True)
                    
                    for idx in significant_indices:
                        p_val = probs[idx] * 100
                        role_info_list.append(f"P{idx+1}({p_val:.0f}%)")

                # 拼接显示，例如：Pudge <br> P4(45%) P2(30%) <br> (+0.02)
                role_str = f"<br><span style='font-size:10px; color:gray;'>{' '.join(role_info_list)}</span>"
                hero_labels.append(f"{name}{role_str}<br>({d_val:+.2f})")

            # 4. 绘制交互式热力图
            fig = px.imshow(
                matrix,
                x=hero_labels,
                y=hero_labels,
                labels=dict(x=t("heatmap_target"), y=t("heatmap_source"), color=t("heatmap_attn")),
                color_continuous_scale="RdBu_r", # 红蓝配色，深色代表强关注
                text_auto=".1f",
                aspect="auto"
            )

            # 5. 美化布局：增加十字分割线区分敌我
            fig.add_shape(type="line", x0=4.5, y0=-0.5, x1=4.5, y1=9.5, line=dict(color="white", width=3))
            fig.add_shape(type="line", x0=-0.5, y0=4.5, x1=9.5, y1=4.5, line=dict(color="white", width=3))
            
            fig.update_layout(
                title=t("heatmap_title"),
                xaxis_title=t("heatmap_x"),
                yaxis_title=t("heatmap_y"),
                height=700
            )

            st.plotly_chart(fig, use_container_width=True)
            
            st.info(t("matrix_read", prob=base_prob))

        except Exception as e:
            st.error(t("matrix_fail", err=e))
    else:
        st.warning(t("xgb_no_matrix"))