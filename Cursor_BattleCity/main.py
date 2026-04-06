"""坦克大战入口：在项目根目录执行 python main.py"""

import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from src.game import run

if __name__ == "__main__":
    run()
