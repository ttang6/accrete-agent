"""顶层 conftest.py —— 仅为把仓库根挂进 pytest 的 sys.path。

为什么需要：
- 项目用 src/ layout：`nanoagent` 通过 setuptools editable install 进入 sys.path
- 但 `evals/` 是 dev tool（eval harness），故意不在 src/ 也不入安装包
- pytest 默认不把项目根加进 sys.path → `from evals.grader import ...` 失败

pytest 启动 collection 时向上扫 conftest.py，每找到一个就把它所在目录加进
sys.path。空文件即可达成此效果，是 pytest 文档/sklearn/pandas 等使用的惯用做法。
比 `[tool.pytest.ini_options] pythonpath = ["."]` 更早期 hook、更可靠。

不要在这里堆 fixtures——共享 fixture 应该按测试范围放对应子目录的 conftest.py
（如 test/runtime_memory/conftest.py 已存在）。本文件保持空仅作 sys.path 锚点。
"""
