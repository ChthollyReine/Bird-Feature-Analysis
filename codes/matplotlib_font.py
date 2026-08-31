"""Matplotlib 中文字体配置：注册项目内置字体，确保本地和 Streamlit Cloud 均正常显示中文。

在 import matplotlib.pyplot 之前导入本模块即可：
    from . import matplotlib_font
    import matplotlib.pyplot as plt
"""

import matplotlib
import matplotlib.font_manager as fm

# 项目内置中文字体路径
_FONT_DIR = __import__("pathlib").Path(__file__).resolve().parent / "fonts"
_FONT_PATH = str(_FONT_DIR / "simhei.ttf")

# 注册字体到 matplotlib，并设为默认 sans-serif 字体
fm.fontManager.addfont(_FONT_PATH)
matplotlib.rcParams["font.sans-serif"] = ["SimHei"] + matplotlib.rcParams["font.sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False