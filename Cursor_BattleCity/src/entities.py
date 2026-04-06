"""坦克、子弹等实体。"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List

import pygame

from . import config as C


class Dir(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


def dir_unit(d: Dir) -> tuple[int, int]:
    if d == Dir.UP:
        return 0, -1
    if d == Dir.RIGHT:
        return 1, 0
    if d == Dir.DOWN:
        return 0, 1
    return -1, 0


@dataclass
class Bullet:
    x: float
    y: float
    direction: Dir
    from_player: bool
    alive: bool = True

    def rect(self) -> pygame.Rect:
        s = C.BULLET_SIZE
        return pygame.Rect(int(self.x - s / 2), int(self.y - s / 2), s, s)


@dataclass
class Tank:
    x: float
    y: float
    direction: Dir
    is_player: bool
    speed: float
    alive: bool = True
    shoot_cd: float = 0.0
    ai_turn_cd: float = 0.0
    ai_shoot_cd: float = field(default_factory=lambda: random.uniform(C.ENEMY_SHOOT_MIN, C.ENEMY_SHOOT_MAX))
    # 玩家拾取道具后，每次开火子弹数 = 1 + shot_bonus
    shot_bonus: int = 0

    def rect(self) -> pygame.Rect:
        s = C.TANK_SIZE
        return pygame.Rect(int(self.x - s / 2), int(self.y - s / 2), s, s)


@dataclass
class Explosion:
    x: float
    y: float
    t: float = 0.0
    duration: float = 0.35

    def alive(self) -> bool:
        return self.t < self.duration


def try_move_tank(
    tank: Tank,
    dx: float,
    dy: float,
    grid: List[List[int]],
    others: List[Tank],
) -> None:
    """与墙体、其它坦克碰撞检测后移动。"""
    r = tank.rect()
    nr = r.move(dx, dy)
    if nr.left < C.MAP_RECT.left or nr.right > C.MAP_RECT.right or nr.top < C.MAP_RECT.top or nr.bottom > C.MAP_RECT.bottom:
        return
    if tank_hits_map(nr, grid):
        return
    for o in others:
        if o is tank or not o.alive:
            continue
        if nr.colliderect(o.rect()):
            return
    tank.x += dx
    tank.y += dy


def tank_hits_map(rect: pygame.Rect, grid: List[List[int]]) -> bool:
    from . import level as L

    c0 = max(0, rect.left // C.TILE)
    c1 = min(C.COLS - 1, rect.right // C.TILE)
    r0 = max(0, rect.top // C.TILE)
    r1 = min(C.ROWS - 1, rect.bottom // C.TILE)
    for row in range(r0, r1 + 1):
        for col in range(c0, c1 + 1):
            t = grid[row][col]
            if L.is_solid_for_tank(t):
                if rect.colliderect(L.tile_rect(col, row)):
                    return True
    return False
