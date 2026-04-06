"""关卡加载与可破坏砖墙。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pygame

from . import config as C


def _default_level_rows() -> List[List[int]]:
    """内置第一关（无文件时回退）。"""
    rows: List[List[int]] = []
    for r in range(C.ROWS):
        row = [C.EMPTY] * C.COLS
        rows.append(row)
    # 外圈钢墙
    for c in range(C.COLS):
        rows[0][c] = C.STEEL
        rows[C.ROWS - 1][c] = C.STEEL
    for r in range(C.ROWS):
        rows[r][0] = C.STEEL
        rows[r][C.COLS - 1] = C.STEEL
    # 中间障碍
    for r in range(4, 10):
        for c in range(6, 11):
            rows[r][c] = C.BRICK
        for c in range(14, 19):
            rows[r][c] = C.BRICK
    for c in range(10, 15):
        rows[8][c] = C.STEEL
    # 水域示例
    for c in range(3, 7):
        rows[12][c] = C.WATER
    for c in range(18, 22):
        rows[12][c] = C.WATER
    # 底部基地 2×2
    bc = C.COLS // 2 - 1
    br = C.ROWS - 3
    rows[br][bc] = C.BASE
    rows[br][bc + 1] = C.BASE
    rows[br + 1][bc] = C.BASE
    rows[br + 1][bc + 1] = C.BASE
    # 基地周围砖
    for dr, dc in ((-1, -1), (-1, 0), (-1, 1), (-1, 2), (0, -1), (0, 2), (1, -1), (1, 2)):
        r, col = br + dr, bc + dc
        if 0 <= r < C.ROWS and 0 <= col < C.COLS and rows[r][col] == C.EMPTY:
            rows[r][col] = C.BRICK
    # 草地装饰（若干格）
    for c in range(1, 5):
        rows[15][c] = C.GRASS
    for c in range(20, 24):
        rows[15][c] = C.GRASS

    # 道具生成：在空白区域放置增强火力的子弹道具（可拾取）
    # 避免放在基地/出生附近的大面积区域，尽量落在“空地”上
    for r, c in [
        (7, 4),
        (7, 20),
        (9, 3),
        (9, 21),
        (10, 12),
    ]:
        if rows[r][c] == C.EMPTY:
            rows[r][c] = C.POWERUP_BULLET
    return rows


def load_level(path: Path | None = None) -> List[List[int]]:
    """从 JSON 加载二维数组，格式：{ \"rows\": [[...], ...] }，尺寸须与 COLS×ROWS 一致。"""
    if path is not None and path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data["rows"]
        if len(rows) != C.ROWS or any(len(row) != C.COLS for row in rows):
            raise ValueError(f"关卡尺寸须为 {C.ROWS}x{C.COLS}")
        return [list(map(int, row)) for row in rows]
    return _default_level_rows()


def tile_rect(col: int, row: int) -> pygame.Rect:
    return pygame.Rect(col * C.TILE, row * C.TILE, C.TILE, C.TILE)


def is_solid_for_tank(t: int) -> bool:
    return t in (C.BRICK, C.STEEL, C.WATER, C.BASE)


def is_solid_for_bullet(t: int) -> bool:
    """子弹可飞过草地；水域不挡子弹。"""
    return t in (C.BRICK, C.STEEL)


def is_base_tile(t: int) -> bool:
    return t == C.BASE


def damage_brick(grid: List[List[int]], col: int, row: int) -> bool:
    """砖墙被击中变空。返回 True 表示该格为砖且已被打掉。"""
    if not (0 <= row < C.ROWS and 0 <= col < C.COLS):
        return False
    if grid[row][col] == C.BRICK:
        grid[row][col] = C.EMPTY
        return True
    return False
