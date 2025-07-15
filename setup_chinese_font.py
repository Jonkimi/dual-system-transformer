import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os

def setup_chinese_font():
    """配置中文字体支持"""
    # 字体替代方案（按优先级排序）
    font_options = [
        'SimHei',                  # 首选 - SimHei
        'Noto Sans CJK SC',         # 备用1 - Google Noto
        'WenQuanYi Zen Hei',       # 备用2 - 文泉驿正黑
        'WenQuanYi Micro Hei',     # 备用3 - 文泉驿微米黑
        'Microsoft YaHei',         # 备用4 - 微软雅黑
        'DejaVu Sans',             # 最后的回退
    ]
    
    # 尝试设置字体
    for font_name in font_options:
        try:
            # 查找字体文件
            font_path = fm.findfont(fm.FontProperties(family=font_name))
            
            # 配置为系统默认
            plt.rcParams['font.family'] = font_name
            plt.rcParams['font.sans-serif'] = [font_name]
            plt.rcParams['axes.unicode_minus'] = False
            
            print(f"已成功配置字体: {font_name} ({os.path.basename(font_path)})")
            return
        except:
            continue
    
    # 如果所有字体都失败，使用回退方案
    plt.rcParams['font.family'] = 'sans-serif'
    print("警告: 未找到中文字体，将使用系统默认")

# 在脚本开头调用设置
setup_chinese_font()