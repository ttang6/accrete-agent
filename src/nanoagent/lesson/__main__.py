"""`python -m nanoagent.lesson <subcommand>` 入口薄壳。

控制台 stdout/stderr utf-8 配置只在这里处理——避免污染 pytest capsys。
"""

import sys

from nanoagent.lesson.cli import main

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
