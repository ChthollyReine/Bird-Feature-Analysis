"""鸟类图案生态分析 - Streamlit 图形界面。

功能：
  1. 浏览/检索 CUB 鸟类图像，展示图案特征与 AVONET 生态性状，输出数据集统计。
  2. 上传单张/多张鸟类图像，运行检测与特征提取，自动识别最近物种并给出生态指标。

运行：
    python -m streamlit run app.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
import streamlit as st

# 保证以项目根目录运行时能导入 codes 包
sys.path.insert(0, str(Path(__file__).resolve().parent))

from codes import config, data_loader, preprocessing, segmentation, ecological, ecological_analysis  # noqa: E402
from codes.features import extract_all  # noqa: E402

st.set_page_config(page_title="鸟类图案生态分析", page_icon="🐦", layout="wide")


# ---------------------------------------------------------------- 中文对照
# AVONET 类别性状取值 -> 中文
TRAIT_CN = {
    "Habitat": {
        "Coastal": "沿海", "Desert": "荒漠", "Forest": "森林",
        "Grassland": "草原", "Human Modified": "人工改造", "Marine": "海洋",
        "Riverine": "河流", "Rock": "岩石", "Shrubland": "灌木丛",
        "Wetland": "湿地", "Woodland": "林地",
    },
    "Trophic.Niche": {
        "Aquatic predator": "水生捕食者", "Frugivore": "食果", "Granivore": "食谷",
        "Herbivore aquatic": "水生植食", "Herbivore terrestrial": "陆生植食",
        "Invertivore": "食无脊椎动物", "Nectarivore": "食蜜", "Omnivore": "杂食",
        "Scavenger": "食腐", "Vertivore": "食脊椎动物",
    },
    "Primary.Lifestyle": {
        "Aerial": "空中", "Aquatic": "水生", "Generalist": "泛化",
        "Insessorial": "栖枝", "Terrestrial": "陆生",
    },
    "Trophic.Level": {
        "Carnivore": "肉食", "Herbivore": "植食", "Omnivore": "杂食", "Scavenger": "食腐",
    },
    "Migration": {"1": "留鸟(不迁徙)", "2": "部分迁徙", "3": "完全迁徙"},
}


def translate(col, val):
    """把类别性状取值翻译为中文；非类别或未知值原样返回。"""
    if pd.isna(val):
        return "—"
    key = str(val)
    if col == "Migration":
        try:
            key = str(int(float(val)))
        except (ValueError, TypeError):
            pass
    return TRAIT_CN.get(col, {}).get(key, val)


# ---------------------------------------------------------------- 缓存
@st.cache_resource
def get_dataset():
    # 优先用完整数据集，否则回退到精简 sample_images.zip（云端部署用）
    for path in (config.DATA_ZIP, config.SAMPLE_ZIP):
        if path.exists():
            return data_loader.CUBDataset(path)
    return None


@st.cache_data
def load_library():
    df, feature_cols, scaler = ecological_analysis.load_feature_library()
    df["中文名"] = df["cub_name"].map(ecological.cn_name)
    return df, feature_cols, scaler


@st.cache_data
def class_first_image():
    ds = get_dataset()
    if ds is None:
        return {}
    first = {}
    for img_id, label in ds.id2label.items():
        first.setdefault(label, img_id)
    return first


@st.cache_data
def dataset_stats(df):
    def _cn(s, col):
        s = s.copy()
        s.index = [translate(col, x) for x in s.index]
        return s

    return {
        "species": int(df["class_id"].nunique()),
        "habitat": _cn(df["Habitat"].value_counts(dropna=True), "Habitat"),
        "migration": _cn(df["Migration"].value_counts(dropna=True), "Migration"),
        "niche": _cn(df["Trophic.Niche"].value_counts(dropna=True), "Trophic.Niche"),
    }


# ---------------------------------------------------------------- 处理函数
def process_image(file_bytes):
    """对上传图片运行 预处理 -> 分割 -> 特征提取，返回各中间结果。"""
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = preprocessing.preprocess(img)
    mask, masked = segmentation.segment(img)
    feats = extract_all(img, mask=mask)
    return img, mask, masked, feats


def to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# 关键图案指标（界面展示用）
KEY_METRICS = [
    ("左右对称性", "sym_lr"),
    ("上下对称性", "sym_tb"),
    ("分形维数(复杂度)", "fractal_dim"),
    ("纹理对比度", "glcm_contrast"),
    ("纹理同质性", "glcm_homogeneity"),
    ("纹理熵", "glcm_entropy"),
    ("颜色饱和度均值", "color_s_mean"),
    ("轮廓圆度", "shape_circularity"),
]

# 图案特征含义说明（悬停指标可见，也在"机理解释"页展示）
FEATURE_DESC = {
    "sym_lr": "左右对称性：鸟体左右两半图案的相似度（-1~1，越大越对称）。对称性常与发育稳定性、个体健康质量相关。",
    "sym_tb": "上下对称性：鸟体上下两半图案的相似度。",
    "fractal_dim": "分形维数：图案的复杂度/不规则度，越大图案越复杂。高复杂度可能与伪装、性选择信号有关。",
    "glcm_contrast": "纹理对比度：局部灰度差异，越大羽毛斑点/条纹越锐利明显。",
    "glcm_homogeneity": "纹理同质性：灰度分布是否均匀，越大纹理越平滑一致。",
    "glcm_entropy": "纹理熵：纹理的随机性/信息量，越大羽毛图案越杂乱、信息越丰富。",
    "color_s_mean": "颜色饱和度均值：羽毛色彩鲜艳程度，越大越鲜艳（常与性选择、物种识别信号相关）。",
    "shape_circularity": "轮廓圆度：鸟体轮廓接近圆形的程度（0~1，越接近 1 越圆）。",
}

# AVONET 生态性状（label, 字段, 单位, 说明）
TRAIT_DISPLAY = [
    ("体重", "Mass", "g", "平均体重"),
    ("喙长", "Beak.Length_Culmen", "mm", "平均上喙长度"),
    ("翅长", "Wing.Length", "mm", "平均翅长"),
    ("栖息地", "Habitat", "", "主要栖息地类型"),
    ("迁徙等级", "Migration", "", "迁徙程度"),
    ("食性", "Trophic.Niche", "", "主要食性类型"),
    ("生活方式", "Primary.Lifestyle", "", "主要生活方式"),
    ("分布范围", "Range.Size", "km²", "地理分布范围总面积"),
]


def render_traits(row):
    """用 st.metric 展示某物种的生态性状"""
    cols = st.columns(4)
    for i, (label, key, unit, desc) in enumerate(TRAIT_DISPLAY):
        raw = row.get(key)
        if key in TRAIT_CN:
            val = translate(key, raw)
        elif pd.isna(raw):
            val = "—"
        elif isinstance(raw, float):
            if unit == "km²":
                val = f"{raw:,.0f} {unit}"
            else:
                val = f"{raw:.1f} {unit}"
        else:
            val = raw
        cols[i % 4].metric(label, val, help=desc)


def render_metrics(feats):
    """用 st.metric 展示图案评价指标"""
    cols = st.columns(4)
    for i, (label, key) in enumerate(KEY_METRICS):
        val = feats.get(key, 0.0)
        cols[i % 4].metric(label, f"{val:.3f}", help=FEATURE_DESC.get(key, ""))


# ---------------------------------------------------------------- 页面
def page_browse(df, feature_cols):
    st.header("🐦 数据集浏览与统计")

    # ---- 数据集统计
    st.subheader("数据集统计")
    stats = dataset_stats(df)
    c1, c2, c3 = st.columns(3)
    c1.metric("鸟类物种数", stats["species"])
    c2.metric("图片总数", 11788)
    c3.metric("平均每类图片", f"{11788 // stats['species']}")

    a1, a2, a3 = st.columns(3)
    a1.markdown("**栖息地分布**")
    a1.bar_chart(stats["habitat"])
    a2.markdown("**迁徙等级分布**")
    a2.bar_chart(stats["migration"])
    a3.markdown("**食性分布**")
    a3.bar_chart(stats["niche"])

    st.divider()

    # ---- 浏览 / 检索
    st.subheader("浏览与检索")
    query = st.text_input("按中文名 / 俗名 / 学名检索", "")
    all_rows = df.copy()
    if query.strip():
        q = query.strip().lower()
        mask = all_rows["cub_name"].str.lower().str.contains(q, na=False) | \
               all_rows["scientific_name"].str.lower().str.contains(q, na=False) | \
               all_rows["中文名"].str.contains(q, na=False)
        all_rows = all_rows[mask]

    if all_rows.empty:
        st.warning("未找到匹配的物种。")
        return

    name_map = {f"{r['中文名']}（{r['cub_name']}）": r["class_id"]
                for _, r in all_rows.iterrows()}
    choice = st.selectbox("选择物种", list(name_map.keys()))
    cid = name_map[choice]
    row = df[df["class_id"] == cid].iloc[0]

    col_img, col_info = st.columns([1, 2])
    with col_img:
        ds = get_dataset()
        img_id = class_first_image().get(int(cid))
        if ds is not None and img_id is not None:
            img = ds.load_image(img_id)
            st.image(to_rgb(img), caption=row["中文名"], use_container_width=True)
        else:
            st.info("未提供原始数据集（archive.zip），无法显示缩略图")
    with col_info:
        st.markdown(f"**中文名**：{row['中文名']}")
        st.markdown(f"**俗名**：{row['cub_name']}")
        st.markdown(f"**学名**：{row['scientific_name']}")
        st.markdown("**生态性状（AVONET）**")
        render_traits(row)

    st.markdown("**图案特征**")
    render_metrics({k: float(row[k]) for k in [m[1] for m in KEY_METRICS] if k in row})


def page_analyze(df, feature_cols, scaler):
    st.header("🖼️ 图像分析与物种识别")
    st.caption("上传鸟类图像，自动运行 预处理 → 分割 → 特征提取，并匹配最相似物种输出生态指标。")

    files = st.file_uploader("选择鸟类图像", type=["jpg", "jpeg", "png"],
                             accept_multiple_files=True)

    if not files:
        st.info("请上传一张或多张鸟类图像。")
        return

    for f in files:
        st.divider()
        st.subheader(f"📷 {f.name}")
        res = process_image(f.getvalue())
        if res is None:
            st.error(f"无法读取图像 {f.name}")
            continue

        img, mask, masked, feats = res

        c1, c2, c3 = st.columns(3)
        c1.image(to_rgb(img), caption="预处理后图像", use_container_width=True)
        c2.image(mask, caption="分割掩膜", use_container_width=True, clamp=True)
        c3.image(to_rgb(masked), caption="检测主体", use_container_width=True)

        st.markdown("**图案评价指标**")
        render_metrics(feats)

        st.markdown("**最近物种匹配**")
        top = ecological_analysis.predict_species(feats, df, feature_cols, scaler, top_k=3)
        best = top.iloc[0]
        st.success(f"识别结果：{best['中文名']}（{best['cub_name']} / {best['scientific_name']}），距离 {best['distance']:.3f}")
        st.markdown("**该物种生态指标（AVONET）**")
        render_traits(best)

        st.markdown("**Top-3 候选**")
        st.dataframe(top[["class_id", "中文名", "cub_name", "scientific_name", "distance"]],
                     use_container_width=True)


# ---------------------------------------------------------------- 机理解释
CONTINUOUS_TRAITS = [
    "Mass", "Beak.Length_Culmen", "Beak.Width", "Beak.Depth",
    "Tarsus.Length", "Wing.Length", "Hand-Wing.Index", "Tail.Length",
    "Migration", "Range.Size", "Centroid.Latitude",
]

TRAIT_LABEL = {t[1]: t[0] for t in TRAIT_DISPLAY}
# 补充相关性热图中出现的其余性状的中文名
TRAIT_LABEL.update({
    "Beak.Width": "喙宽",
    "Beak.Depth": "喙高",
    "Tarsus.Length": "跗跖长",
    "Hand-Wing.Index": "手翼指数(HWI)",
    "Tail.Length": "尾长",
    "Centroid.Latitude": "分布中心纬度",
})
FEATURE_LABEL = {m[1]: m[0] for m in KEY_METRICS}


def compute_correlations(df, feature_cols):
    """关键图案特征 X 连续性状 的 Spearman 相关，返回按 p 排序的长表"""
    from scipy import stats
    feats = [m[1] for m in KEY_METRICS if m[1] in feature_cols]
    rows = []
    for trait in CONTINUOUS_TRAITS:
        if trait not in df.columns:
            continue
        tv = pd.to_numeric(df[trait], errors="coerce")
        for f in feats:
            fv = pd.to_numeric(df[f], errors="coerce")
            mask = tv.notna() & fv.notna()
            if mask.sum() < 10:
                continue
            if np.std(tv[mask]) < 1e-12 or np.std(fv[mask]) < 1e-12:
                continue
            rho, p = stats.spearmanr(tv[mask], fv[mask])
            rows.append({"feature": f, "trait": trait, "rho": float(rho),
                         "p": float(p), "n": int(mask.sum())})
    return pd.DataFrame(rows).sort_values("p")



def plot_corr_heatmap(df, feature_cols):
    """关键特征 X 连续性状 的 Spearman 相关热图，返回 matplotlib Figure"""
    from scipy import stats
    feats = [m[1] for m in KEY_METRICS if m[1] in feature_cols]
    traits = [t for t in CONTINUOUS_TRAITS if t in df.columns]
    mat = np.full((len(feats), len(traits)), np.nan)
    for i, f in enumerate(feats):
        for j, t in enumerate(traits):
            tv = pd.to_numeric(df[t], errors="coerce")
            fv = pd.to_numeric(df[f], errors="coerce")
            mask = tv.notna() & fv.notna()
            if mask.sum() >= 10:
                mat[i, j] = stats.spearmanr(tv[mask], fv[mask])[0]
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(traits)))
    ax.set_xticklabels([TRAIT_LABEL.get(t, t) for t in traits], rotation=40, ha="right")
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels([FEATURE_LABEL.get(f, f) for f in feats])
    plt.colorbar(im, ax=ax, label="Spearman rho")
    ax.set_title("图案特征 X 生态性状（Spearman 相关）")
    fig.tight_layout()
    return fig


def page_interpretation(df, feature_cols):
    st.header("🔬 生态指标映射与机理解释")
    st.caption("把 200 种鸟类的图案特征（对称性/复杂度/纹理/颜色/形貌）与 AVONET 生态性状做 Spearman 相关分析，"
               "用于探讨体表图案与环境适应性的关系。注意：相关性不代表因果，结论仅供机理探讨参考。")

    st.subheader("一、图案特征与生态性状的相关性")
    corr = compute_correlations(df, feature_cols)
    sig = corr[corr["p"] < 0.05]
    c1, c2 = st.columns(2)
    c1.metric("显著相关对（p<0.05）", f"{len(sig)} / {len(corr)}")
    c2.metric("分析物种数", int(df["class_id"].nunique()))

    st.markdown("**主要发现（按显著性排序）**")
    for _, r in sig.head(10).iterrows():
        direction = "正相关" if r["rho"] > 0 else "负相关"
        trait = TRAIT_LABEL.get(r["trait"], r["trait"])
        feat = FEATURE_LABEL.get(r["feature"], r["feature"])
        st.markdown(
            f"- **{trait}** 与 **{feat}** 呈 **{direction}**（ρ={r['rho']:.2f}，p={r['p']:.2e}，n={r['n']}）"
        )

    st.markdown("**相关性热图**（红=正相关，蓝=负相关）")
    st.pyplot(plot_corr_heatmap(df, feature_cols))

    st.subheader("二、图案特征的含义")
    for label, key in KEY_METRICS:
        st.markdown(f"- **{label}**：{FEATURE_DESC.get(key, '')}")

    st.subheader("三、生态性状的含义与单位")
    for label, key, unit, desc in TRAIT_DISPLAY:
        u = f"（{unit}）" if unit else ""
        st.markdown(f"- **{label}**{u}：{desc}")


# ---------------------------------------------------------------- 主入口
def main():
    try:
        df, feature_cols, scaler = load_library()
    except FileNotFoundError:
        st.error("未找到物种特征库。请先运行：`python -m codes.ecological_analysis`")
        st.stop()

    st.sidebar.title("导航")
    page = st.sidebar.radio("选择页面", ["数据集浏览与统计", "图像分析与识别", "生态机理解释"])

    if page == "数据集浏览与统计":
        page_browse(df, feature_cols)
    elif page == "图像分析与识别":
        page_analyze(df, feature_cols, scaler)
    else:
        page_interpretation(df, feature_cols)


main()
