"""
项目级常量与路径配置.

统一管理 PROJECT_ROOT 和资源路径，避免各模块重复拼接。
"""

from pathlib import Path

# 项目根目录（source/ 的上两级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

ASSETS_DIR   = PROJECT_ROOT / "assets"
TEXTURES_DIR = ASSETS_DIR / "textures"


class DefaultTextures:
    """默认纹理路径（以字符串形式提供，便于直接赋给 MuJoCo）."""
    FLOOR         = str(TEXTURES_DIR / "light-gray-floor-tile.png")
    WALL          = str(TEXTURES_DIR / "light-gray-plaster.png")
    TABLE_SURFACE = str(TEXTURES_DIR / "ceramic.png")
    TABLE_LEG     = str(TEXTURES_DIR / "metal.png")