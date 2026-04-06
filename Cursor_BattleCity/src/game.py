"""游戏主循环、状态机、碰撞与绘制。"""

from __future__ import annotations

import copy
import random
from enum import Enum, auto
from pathlib import Path
from typing import List, Optional

import pygame

from . import config as C
from . import entities as E
from . import level as L
from .entities import Bullet, Dir, Explosion, Tank, dir_unit, try_move_tank


class GameState(Enum):
    MENU = auto()
    PLAYING = auto()
    PAUSED = auto()
    WIN = auto()
    LOSE = auto()


def _level_path() -> Path:
    return Path(__file__).resolve().parent / "levels" / "level01.json"


def _find_player_spawn(
    grid: List[List[int]],
    enemies: Optional[List[Tank]] = None,
) -> tuple[float, float]:
    enemies = [e for e in (enemies or []) if e.alive]
    for row in range(12, 17):
        for col in range(6, C.COLS - 6):
            if grid[row][col] != C.EMPTY:
                continue
            cx = col * C.TILE + C.TILE / 2
            cy = row * C.TILE + C.TILE / 2
            t = Tank(cx, cy, Dir.UP, True, C.PLAYER_SPEED)
            # 重生不仅要避开地图障碍，还要避开当前屏幕内的敌方坦克
            r = t.rect()
            if not E.tank_hits_map(r, grid) and not any(r.colliderect(e.rect()) for e in enemies):
                return cx, cy
    # 兜底：在地图更大范围随机找空位，尽量避免再次“重叠卡住”
    for _ in range(200):
        row = random.randint(1, C.ROWS - 2)
        col = random.randint(1, C.COLS - 2)
        if grid[row][col] != C.EMPTY:
            continue
        cx = col * C.TILE + C.TILE / 2
        cy = row * C.TILE + C.TILE / 2
        t = Tank(cx, cy, Dir.UP, True, C.PLAYER_SPEED)
        r = t.rect()
        if not E.tank_hits_map(r, grid) and not any(r.colliderect(e.rect()) for e in enemies):
            return cx, cy
    # 最后兜底（理论上很少走到）：给一个固定位置
    return C.COLS // 2 * C.TILE, (C.ROWS - 5) * C.TILE + C.TILE / 2


def _can_spawn_enemy(grid: List[List[int]], cx: float, cy: float) -> bool:
    t = Tank(cx, cy, Dir.DOWN, False, C.ENEMY_SPEED)
    return not E.tank_hits_map(t.rect(), grid)


def _random_enemy_spawn(grid: List[List[int]], existing: List[Tank], player: Optional[Tank]) -> Optional[tuple[float, float]]:
    for _ in range(80):
        col = random.randint(1, C.COLS - 2)
        row = random.randint(1, 4)
        cx = col * C.TILE + C.TILE / 2
        cy = row * C.TILE + C.TILE / 2
        if not _can_spawn_enemy(grid, cx, cy):
            continue
        t = Tank(cx, cy, Dir.DOWN, False, C.ENEMY_SPEED)
        r = t.rect()
        if player and player.alive and r.colliderect(player.rect()):
            continue
        for e in existing:
            if e.alive and r.colliderect(e.rect()):
                break
        else:
            return cx, cy
    return None


def _safe_sys_font(size: int) -> pygame.font.Font:
    """
    可靠地获取一个“能显示中文”的字体。

    - 优先直接加载 Windows 字体文件（避免 SysFont 在部分环境枚举字体时报错）
    - 其次尝试 SysFont（兼容非 Windows）
    - 最后回退到默认字体（可能不含中文）
    """
    # 1) Windows 常见字体文件（大多数机器存在）
    win_font_paths = [
        r"C:\Windows\Fonts\msyh.ttc",   # 微软雅黑
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",  # 黑体
        r"C:\Windows\Fonts\simsun.ttc",  # 宋体
        r"C:\Windows\Fonts\msjhl.ttc",   # 微软正黑体（部分系统）
    ]
    for p in win_font_paths:
        try:
            return pygame.font.Font(p, size)
        except Exception:
            continue

    # 2) SysFont 名称（可能触发字体枚举；用 try/except 兜底）
    for name in ("microsoftyahei", "simhei", "simsun", "arial", "sans"):
        try:
            return pygame.font.SysFont(name, size)
        except Exception:
            continue

    # 3) 最后兜底：默认字体
    return pygame.font.Font(None, size)


class Game:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("坦克大战")
        self.screen = pygame.display.set_mode((C.SCREEN_W, C.SCREEN_H))
        self.clock = pygame.time.Clock()
        self.font = _safe_sys_font(22)
        self.font_big = _safe_sys_font(36)
        self.state = GameState.MENU
        self.grid: List[List[int]] = []
        self.player: Optional[Tank] = None
        self.enemies: List[Tank] = []
        self.bullets: List[Bullet] = []
        self.explosions: List[Explosion] = []
        self.lives = C.PLAYER_LIVES
        self.score = 0
        self.spawned = 0
        self.killed = 0
        self.spawn_timer = 2.0
        self.base_destroyed = False
        self.invuln = 0.0
    def reset(self) -> None:
        self.grid = copy.deepcopy(L.load_level(_level_path()))
        self.bullets.clear()
        self.enemies.clear()
        self.explosions.clear()
        px, py = _find_player_spawn(self.grid, [])
        self.player = Tank(px, py, Dir.UP, True, C.PLAYER_SPEED)
        self.lives = C.PLAYER_LIVES
        self.score = 0
        self.spawned = 0
        self.killed = 0
        self.spawn_timer = 2.0
        self.base_destroyed = False
        self.invuln = 2.0

    def run(self) -> None:
        running = True
        while running:
            dt = self.clock.tick(C.FPS) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    self._on_keydown(event.key)
            keys = pygame.key.get_pressed()
            if self.state == GameState.PLAYING:
                self._handle_hold(keys, dt)

            if self.state == GameState.PLAYING:
                self._update(dt)

            self._draw()
            pygame.display.flip()

        pygame.quit()

    def _on_keydown(self, key: int) -> None:
        if key == pygame.K_ESCAPE:
            if self.state == GameState.PLAYING:
                self.state = GameState.PAUSED
            elif self.state == GameState.PAUSED:
                self.state = GameState.PLAYING
            return

        if self.state == GameState.MENU:
            if key in (pygame.K_SPACE, pygame.K_RETURN):
                self.reset()
                self.state = GameState.PLAYING
            return

        if self.state in (GameState.WIN, GameState.LOSE):
            if key in (pygame.K_SPACE, pygame.K_RETURN, pygame.K_r):
                self.reset()
                self.state = GameState.PLAYING
            elif key == pygame.K_m:
                self.state = GameState.MENU
            return

    def _handle_hold(self, keys, dt: float) -> None:
        if self.state != GameState.PLAYING or self.player is None or not self.player.alive:
            return
        p = self.player
        dx = dy = 0.0
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            dy -= 1.0
            p.direction = Dir.UP
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            dy += 1.0
            p.direction = Dir.DOWN
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            dx -= 1.0
            p.direction = Dir.LEFT
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            dx += 1.0
            p.direction = Dir.RIGHT

        if dx != 0 or dy != 0:
            ln = (dx * dx + dy * dy) ** 0.5
            dx, dy = dx / ln * p.speed * dt, dy / ln * p.speed * dt
            others = [e for e in self.enemies if e.alive]
            try_move_tank(p, dx, dy, self.grid, others)
            # 移动后拾取道具（火力升级）
            self._try_pickup_powerups(p)

        p.shoot_cd = max(0.0, p.shoot_cd - dt)
        shoot = keys[pygame.K_SPACE] or keys[pygame.K_j] or keys[pygame.K_x]
        if shoot and p.shoot_cd <= 0:
            self._fire_bullet(p, True)
            p.shoot_cd = C.PLAYER_SHOOT_COOLDOWN

    def _fire_bullet(self, tank: Tank, from_player: bool) -> None:
        ux, uy = dir_unit(tank.direction)
        off = C.TANK_SIZE / 2 + C.BULLET_SIZE / 2 + 2

        # 玩家拾取火力道具后：每次开火多 1 发
        bonus = min(tank.shot_bonus, C.BULLET_BONUS_MAX) if from_player else 0
        bullet_n = 1 + bonus

        base_x = tank.x + ux * off
        base_y = tank.y + uy * off

        # 在“与方向垂直”的轴上并排生成多发子弹
        gap = C.BULLET_MULTI_GAP
        for i in range(bullet_n):
            offset = (i - (bullet_n - 1) / 2.0) * gap
            if ux != 0:
                # 左右移动：在 y 方向并排
                bx, by = base_x, base_y + offset
            else:
                # 上下移动：在 x 方向并排
                bx, by = base_x + offset, base_y
            self.bullets.append(Bullet(bx, by, tank.direction, from_player))

    def _try_pickup_powerups(self, tank: Tank) -> None:
        """拾取地图上的子弹道具：每拾取一次，下一次开火多 1 发。"""
        if not tank.is_player:
            return
        r = tank.rect()
        c0 = max(0, r.left // C.TILE)
        c1 = min(C.COLS - 1, r.right // C.TILE)
        r0 = max(0, r.top // C.TILE)
        r1 = min(C.ROWS - 1, r.bottom // C.TILE)

        for row in range(r0, r1 + 1):
            for col in range(c0, c1 + 1):
                if self.grid[row][col] != C.POWERUP_BULLET:
                    continue
                tr = L.tile_rect(col, row)
                if r.colliderect(tr):
                    self.grid[row][col] = C.EMPTY
                    tank.shot_bonus = min(C.BULLET_BONUS_MAX, tank.shot_bonus + 1)

    def _update(self, dt: float) -> None:
        assert self.player is not None
        self.invuln = max(0.0, self.invuln - dt)

        # 敌方生成
        if self.spawned < C.ENEMY_WAVE_TOTAL:
            self.spawn_timer -= dt
            if self.spawn_timer <= 0:
                alive_n = sum(1 for e in self.enemies if e.alive)
                if alive_n < C.MAX_ENEMIES_ON_FIELD:
                    pos = _random_enemy_spawn(self.grid, self.enemies, self.player)
                    if pos:
                        cx, cy = pos
                        self.enemies.append(Tank(cx, cy, random.choice(list(Dir)), False, C.ENEMY_SPEED))
                        self.spawned += 1
                self.spawn_timer = C.ENEMY_SPAWN_INTERVAL

        # 敌方 AI
        for e in self.enemies:
            if not e.alive:
                continue
            e.ai_turn_cd -= dt
            e.ai_shoot_cd -= dt
            if e.ai_turn_cd <= 0:
                e.ai_turn_cd = random.uniform(0.4, 1.1)
                if self.player and self.player.alive and random.random() < 0.55:
                    dx = self.player.x - e.x
                    dy = self.player.y - e.y
                    if abs(dx) > abs(dy):
                        e.direction = Dir.RIGHT if dx > 0 else Dir.LEFT
                    else:
                        e.direction = Dir.DOWN if dy > 0 else Dir.UP
                else:
                    e.direction = random.choice(list(Dir))
            ux, uy = dir_unit(e.direction)
            try_move_tank(e, ux * e.speed * dt, uy * e.speed * dt, self.grid, [self.player] + [x for x in self.enemies if x is not e])
            if e.ai_shoot_cd <= 0:
                self._fire_bullet(e, False)
                e.ai_shoot_cd = random.uniform(C.ENEMY_SHOOT_MIN, C.ENEMY_SHOOT_MAX)

        self._update_bullets(dt)

        # 爆炸动画
        for ex in self.explosions:
            ex.t += dt
        self.explosions = [x for x in self.explosions if x.alive()]

        # 胜负
        if self.base_destroyed:
            self.state = GameState.LOSE
        elif self.killed >= C.ENEMY_WAVE_TOTAL:
            self.state = GameState.WIN
        elif self.player and not self.player.alive and self.lives <= 0:
            self.state = GameState.LOSE

    def _bullet_map_collision(self, br: pygame.Rect) -> bool:
        c0 = max(0, br.left // C.TILE)
        c1 = min(C.COLS - 1, br.right // C.TILE)
        r0 = max(0, br.top // C.TILE)
        r1 = min(C.ROWS - 1, br.bottom // C.TILE)
        for row in range(r0, r1 + 1):
            for col in range(c0, c1 + 1):
                t = self.grid[row][col]
                tr = L.tile_rect(col, row)
                if not br.colliderect(tr):
                    continue
                if L.is_base_tile(t):
                    self.base_destroyed = True
                    return True
                if t == C.STEEL:
                    return True
                if t == C.BRICK:
                    L.damage_brick(self.grid, col, row)
                    return True
        return False

    def _bullet_tank_collision(self, b: Bullet, br: pygame.Rect) -> bool:
        if b.from_player:
            for e in self.enemies:
                if e.alive and br.colliderect(e.rect()):
                    e.alive = False
                    self.killed += 1
                    self.score += 100
                    self.explosions.append(Explosion(e.x, e.y))
                    return True
            return False

        if self.player and self.player.alive and self.invuln <= 0 and br.colliderect(self.player.rect()):
            self.player.alive = False
            self.lives -= 1
            self.explosions.append(Explosion(self.player.x, self.player.y))
            if self.lives > 0:
                px, py = _find_player_spawn(self.grid, [e for e in self.enemies if e.alive])
                self.player.x, self.player.y = px, py
                self.player.alive = True
                self.invuln = 2.5
            return True

        for e in self.enemies:
            if e.alive and br.colliderect(e.rect()):
                return True
        return False

    def _update_bullets(self, dt: float) -> None:
        remain: List[Bullet] = []
        travel = C.BULLET_SPEED * dt
        step = 10.0

        for b in self.bullets:
            ux, uy = dir_unit(b.direction)
            dist = travel
            hit_stop = False
            while dist > 0 and not hit_stop:
                d = min(step, dist)
                b.x += ux * d
                b.y += uy * d
                dist -= d
                br = b.rect()
                if not C.MAP_RECT.colliderect(br):
                    hit_stop = True
                    break
                if self._bullet_map_collision(br):
                    hit_stop = True
                    break
                if self._bullet_tank_collision(b, br):
                    hit_stop = True
                    break
            if not hit_stop:
                remain.append(b)
        self.bullets = remain

    def _draw(self) -> None:
        self.screen.fill(C.COLORS["bg"])
        if self.state == GameState.MENU:
            self._draw_menu()
            return

        self._draw_map()
        if self.player and self.player.alive:
            pr = self.player.rect()
            col = C.COLORS["player"]
            if self.invuln > 0 and int(pygame.time.get_ticks() / 120) % 2:
                col = (100, 100, 100)
            pygame.draw.rect(self.screen, col, pr, border_radius=4)
            self._draw_tank_barrel(self.player, col)

        for e in self.enemies:
            if e.alive:
                pygame.draw.rect(self.screen, C.COLORS["enemy"], e.rect(), border_radius=4)
                self._draw_tank_barrel(e, C.COLORS["enemy"])

        for b in self.bullets:
            c = C.COLORS["bullet_p"] if b.from_player else C.COLORS["bullet_e"]
            pygame.draw.rect(self.screen, c, b.rect())

        self._draw_grass_overlay()

        for ex in self.explosions:
            r = int(8 + 22 * (ex.t / ex.duration))
            pygame.draw.circle(self.screen, (255, 200, 80), (int(ex.x), int(ex.y)), r, 2)

        self._draw_hud()

        if self.state == GameState.PAUSED:
            self._draw_overlay("已暂停 · 按 ESC 继续")
        elif self.state == GameState.WIN:
            self._draw_overlay("胜利！得分 %d · 空格或 R 再来 · M 回菜单" % self.score)
        elif self.state == GameState.LOSE:
            reason = "基地被毁" if self.base_destroyed else "生命耗尽"
            self._draw_overlay("失败（%s）· 空格或 R 重试 · M 回菜单" % reason)

    def _draw_tank_barrel(self, tank: Tank, color: tuple[int, int, int]) -> None:
        ux, uy = dir_unit(tank.direction)
        cx, cy = tank.x + ux * (C.TANK_SIZE / 2 + 4), tank.y + uy * (C.TANK_SIZE / 2 + 4)
        pygame.draw.circle(self.screen, color, (int(cx), int(cy)), 4)

    def _draw_map(self) -> None:
        assert self.grid
        for row in range(C.ROWS):
            for col in range(C.COLS):
                t = self.grid[row][col]
                if t == C.GRASS:
                    continue
                r = L.tile_rect(col, row)
                if t == C.EMPTY:
                    continue
                if t == C.BRICK:
                    pygame.draw.rect(self.screen, C.COLORS["brick"], r)
                elif t == C.STEEL:
                    pygame.draw.rect(self.screen, C.COLORS["steel"], r)
                elif t == C.WATER:
                    pygame.draw.rect(self.screen, C.COLORS["water"], r)
                elif t == C.POWERUP_BULLET:
                    # 道具：增强火力（每次开火 +1 发）
                    pygame.draw.circle(self.screen, C.COLORS["powerup"], r.center, 8)
                    pygame.draw.circle(self.screen, C.COLORS["text"], r.center, 8, width=2)
                elif t == C.BASE:
                    if not self.base_destroyed:
                        pygame.draw.rect(self.screen, C.COLORS["base"], r)
                    else:
                        pygame.draw.rect(self.screen, (80, 40, 40), r)

    def _draw_grass_overlay(self) -> None:
        for row in range(C.ROWS):
            for col in range(C.COLS):
                if self.grid[row][col] == C.GRASS:
                    s = pygame.Surface((C.TILE, C.TILE), pygame.SRCALPHA)
                    s.fill((*C.COLORS["grass"], 160))
                    self.screen.blit(s, (col * C.TILE, row * C.TILE))

    def _draw_hud(self) -> None:
        pygame.draw.rect(self.screen, C.COLORS["hud_bg"], C.HUD_RECT)
        shot_n = 1 + self.player.shot_bonus if self.player else 1
        t1 = self.font.render(
            "生命: %d   得分: %d   击毁: %d/%d   火力: %d发/次"
            % (self.lives, self.score, self.killed, C.ENEMY_WAVE_TOTAL, shot_n),
            True,
            C.COLORS["text"],
        )
        self.screen.blit(t1, (16, C.HUD_RECT.y + 12))
        h2 = self.font.render("拾取子弹道具：开火子弹 +1   空格/J 射击 · ESC 暂停", True, (160, 160, 170))
        self.screen.blit(h2, (16, C.HUD_RECT.y + 40))

    def _draw_overlay(self, text: str) -> None:
        ov = pygame.Surface((C.SCREEN_W, C.SCREEN_H), pygame.SRCALPHA)
        ov.fill((*C.COLORS["overlay"], 140))
        self.screen.blit(ov, (0, 0))
        surf = self.font_big.render(text, True, C.COLORS["text"])
        self.screen.blit(surf, surf.get_rect(center=(C.SCREEN_W // 2, C.SCREEN_H // 2)))

    def _draw_menu(self) -> None:
        title = self.font_big.render("坦克大战", True, C.COLORS["text"])
        self.screen.blit(title, title.get_rect(center=(C.SCREEN_W // 2, 220)))
        lines = [
            "空格 / Enter 开始",
            "WASD 或方向键移动 · 空格 / J 射击",
            "消灭全部敌方坦克，保护基地（金色方块）",
            "基地或己方被毁、生命耗尽则失败",
        ]
        y = 300
        for line in lines:
            s = self.font.render(line, True, C.COLORS["text"])
            self.screen.blit(s, s.get_rect(center=(C.SCREEN_W // 2, y)))
            y += 36


def run() -> None:
    Game().run()
