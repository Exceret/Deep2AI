from datetime import datetime
from typing import Literal, Optional

def ts_print(
    message: str,
    symbol: Optional[Literal["info", "success", "warning", "error", "debug"]] = None,
    color: bool = True,
) -> None:
    """
    带时间戳的信息输出函数

    Args:
        message: 要输出的消息内容
        symbol: CLI 符号类型，可选值: 'info', 'success', 'warning', 'error', 'debug'，默认无符号
        color: 是否启用 ANSI 颜色输出（默认 True）
    """
    timestamp = datetime.now().strftime("[%Y/%m/%d %H:%M:%S]")

    symbols: dict = {
        "info": ("ℹ", "\033[36m" if color else ""),  # Cyan
        "success": ("✔", "\033[32m" if color else ""),  # Green
        "warning": ("⚠", "\033[33m" if color else ""),  # Yellow
        "error": ("✖", "\033[31m" if color else ""),  # Red
        "debug": ("◼", "\033[35m" if color else ""),  # Magenta
    }

    t_prefix: str = f"{timestamp} "
    if symbol in symbols:
        sym, col = symbols[symbol]
        symbol_prefix: str = f"{col}{sym} \033[0m" if color else f"{sym} "
    else:
        symbol_prefix: str = ""

    print(symbol_prefix + f"{t_prefix}{message}")
