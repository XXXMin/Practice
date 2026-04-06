"""全局常量与配色。"""

import pygame

# 显示
SCREEN_W = 800
SCREEN_H = 704
MAP_RECT = pygame.Rect(0, 0, 800, 640)
HUD_RECT = pygame.Rect(0, 640, 800, 64)
FPS = 60

# 地图网格（25 列 × 20 行），单格像素边长
COLS = 25
ROWS = 20
TILE = 32

# 坦克与子弹
TANK_SIZE = 28
BULLET_SIZE = 8
PLAYER_SPEED = 120.0
ENEMY_SPEED = 70.0
BULLET_SPEED = 280.0
PLAYER_SHOOT_COOLDOWN = 0.45
ENEMY_SHOOT_MIN = 1.2
ENEMY_SHOOT_MAX = 2.4
PLAYER_LIVES = 3
MAX_ENEMIES_ON_FIELD = 4
ENEMY_WAVE_TOTAL = 8
ENEMY_SPAWN_INTERVAL = 4.0

# 瓦片类型（与关卡 JSON 数字一致）
EMPTY = 0
BRICK = 1
STEEL = 2
WATER = 3
GRASS = 4
BASE = 5
POWERUP_BULLET = 6  # 增强火力：每次开火多 1 发

COLORS = {
    "bg": (40, 40, 48),
    "hud_bg": (28, 28, 36),
    "brick": (180, 90, 50),
    "steel": (140, 140, 150),
    "water": (50, 90, 200),
    "grass": (50, 140, 60),
    "base": (220, 180, 60),
    "powerup": (80, 220, 255),
    "player": (80, 200, 120),
    "enemy": (220, 80, 80),
    "bullet_p": (220, 220, 100),
    "bullet_e": (255, 160, 80),
    "text": (230, 230, 235),
    "overlay": (0, 0, 0),
}

# 火力：基础 1 发，每拾取一次 +1（上限避免无限增长）
BULLET_BONUS_MAX = 4  # 最多变为 1+4 = 5 发/次
BULLET_MULTI_GAP = 8   # 同向多发子弹的并排行距（像素）
