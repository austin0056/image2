"""图标库检索：基于 Iconify 数据 + rapidfuzz 模糊搜索 + 简易中英映射。

用途：根据用户 prompt 找出 top-k 个最相关的图标作为 few-shot 样本，
让 LLM 学习真实设计师作品的风格。
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz, process

log = logging.getLogger("image2.icon_search")

# 库名 → 文件名（不带 .json）
LIBRARY_FILES = {
    "lucide": "lucide",
    "phosphor": "phosphor",
    "heroicons-outline": "heroicons",
    "heroicons-solid": "heroicons",
    "tabler": "tabler",
    # feather/material/duotone/auto 没有 1:1 对应,落到 lucide
    "feather": "lucide",
    "material": "heroicons",
    "duotone": "phosphor",
    "auto": "lucide",
}

# 高频中文关键词 → 英文映射,覆盖 80% 常见图标主题
CN_HINTS: dict[str, str] = {
    # 用户/账号
    "用户": "user", "人": "person", "账户": "account", "账号": "account",
    "头像": "avatar", "登录": "login", "退出": "logout", "管理员": "admin",
    "团队": "team", "群组": "group", "联系人": "contact",
    # 文件/文档
    "文件": "file", "文件夹": "folder", "文档": "document", "图片": "image",
    "照片": "photo", "视频": "video", "音乐": "music", "音频": "audio",
    "下载": "download", "上传": "upload", "导入": "import", "导出": "export",
    "附件": "attachment", "压缩": "archive", "解压": "extract",
    # 操作
    "添加": "add plus", "删除": "delete trash", "编辑": "edit pencil",
    "保存": "save", "复制": "copy", "粘贴": "paste", "剪切": "cut scissors",
    "搜索": "search", "查找": "find search", "刷新": "refresh",
    "同步": "sync refresh", "撤销": "undo", "重做": "redo",
    "锁": "lock", "解锁": "unlock", "钥匙": "key",
    "分享": "share", "发送": "send", "邮件": "mail email",
    # 导航
    "首页": "home", "返回": "back arrow", "前进": "forward",
    "上": "up arrow", "下": "down arrow", "左": "left arrow", "右": "right arrow",
    "菜单": "menu", "更多": "more", "设置": "settings gear",
    "通知": "bell notification", "消息": "message chat",
    "聊天": "chat message", "评论": "comment",
    # 业务
    "购物车": "cart shopping", "购物": "shopping bag", "支付": "payment credit-card",
    "钱包": "wallet", "金钱": "money dollar", "礼物": "gift",
    "标签": "tag label", "书签": "bookmark", "星标": "star favorite",
    "收藏": "heart favorite", "喜欢": "heart like",
    "电话": "phone", "地址": "map-pin location", "位置": "map-pin location",
    "时间": "clock time", "日历": "calendar", "日期": "calendar date",
    "天气": "cloud sun", "温度": "thermometer",
    # 数据/技术
    "数据": "database", "数据库": "database", "服务器": "server",
    "云": "cloud", "云端": "cloud", "网络": "wifi network", "信号": "signal",
    "代码": "code", "终端": "terminal", "命令": "terminal",
    "图表": "chart", "统计": "chart-bar statistics", "趋势": "trending-up",
    # 媒体
    "播放": "play", "暂停": "pause", "停止": "stop", "录制": "record",
    "音量": "volume", "静音": "volume-x mute",
    "摄像头": "camera", "相机": "camera", "麦克风": "microphone",
    # 形状/状态
    "勾选": "check checkmark", "对": "check", "错": "x close",
    "警告": "alert-triangle warning", "信息": "info", "帮助": "help",
    "成功": "check-circle success", "失败": "x-circle fail",
    "圆": "circle", "方": "square", "三角": "triangle", "心": "heart",
    # 其他高频
    "眼睛": "eye", "灯": "lightbulb", "火": "flame", "水": "droplet",
    "电": "zap lightning", "电池": "battery", "插头": "plug",
    "工具": "tool wrench", "齿轮": "gear settings",
    "按钮": "button", "开关": "toggle switch",
    "全屏": "maximize", "缩小": "minimize",
    "缩放": "zoom",
}

# 已加载的图标库缓存:{file_key: {name: {body, width, height}}}
_INDEX: dict[str, dict[str, dict[str, Any]]] = {}


def _data_dir() -> Path:
    return Path(__file__).parent / "icon_data"


def _load(file_key: str) -> dict[str, dict[str, Any]]:
    """加载某个图标库的 JSON 进内存(首次访问时)。"""
    if file_key in _INDEX:
        return _INDEX[file_key]
    path = _data_dir() / f"{file_key}.json"
    if not path.exists():
        log.warning("icon data not found: %s", path)
        _INDEX[file_key] = {}
        return _INDEX[file_key]
    raw = json.loads(path.read_text(encoding="utf-8"))
    default_w = raw.get("width", 24)
    default_h = raw.get("height", 24)
    icons: dict[str, dict[str, Any]] = {}
    for name, info in raw.get("icons", {}).items():
        if not isinstance(info, dict):
            continue
        body = info.get("body")
        if not body:
            continue
        icons[name] = {
            "body": body,
            "width": info.get("width", default_w),
            "height": info.get("height", default_h),
            # 多个别名(便于关键词搜索)
            "aliases": [name] + name.split("-"),
        }
    _INDEX[file_key] = icons
    log.info("loaded %s: %d icons", file_key, len(icons))
    return icons


def translate_query(prompt: str) -> str:
    """把中文 prompt 简易转成英文关键词。
    策略:扫描 CN_HINTS 中的中文词,替换为英文;原文也保留(兼容已经是英文的输入)。
    """
    out_tokens = []
    text = prompt
    for cn, en in CN_HINTS.items():
        if cn in text:
            out_tokens.append(en)
    # 同时保留原文中的英文部分
    en_parts = re.findall(r"[a-zA-Z][a-zA-Z\-]+", prompt)
    out_tokens.extend(en_parts)
    if not out_tokens:
        # 一个中文词都没命中,尝试单字逐个看
        return prompt
    result = " ".join(out_tokens)
    return result


def _wrap_svg(body: str, width: int, height: int, library_hint: str) -> str:
    """把 iconify 的 body 包成完整 SVG 字符串(供作为样本注入 prompt)。"""
    vb = f"0 0 {width} {height}"
    # 大多数库需要 stroke 属性在外层,iconify 的 body 已经写好了相关属性
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vb}" '
        f'fill="none" stroke="currentColor" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round">'
        f"{body}</svg>"
    )


def search_examples(
    prompt: str,
    library: str,
    top_k: int = 5,
    min_score: float = 30.0,
) -> list[dict[str, Any]]:
    """根据 prompt 检索 top-k 相关图标。
    返回:[{name, svg, score}, ...]
    """
    file_key = LIBRARY_FILES.get(library, "lucide")
    icons = _load(file_key)
    if not icons:
        return []

    query = translate_query(prompt).lower()
    if not query.strip():
        return []

    names = list(icons.keys())
    # 用 token_set_ratio:对多关键词友好,词序无关
    matches = process.extract(
        query,
        names,
        scorer=fuzz.token_set_ratio,
        limit=top_k * 2,  # 多取一些,稍后过滤
    )

    out: list[dict[str, Any]] = []
    for name, score, _ in matches:
        if score < min_score:
            continue
        info = icons[name]
        svg = _wrap_svg(info["body"], info["width"], info["height"], library)
        out.append({
            "name": name,
            "svg": svg,
            "score": round(score, 1),
            "library_file": file_key,
        })
        if len(out) >= top_k:
            break
    log.info(
        "search_examples library=%s query=%r → %d hits: %s",
        library, query, len(out), [(o["name"], o["score"]) for o in out],
    )
    return out
