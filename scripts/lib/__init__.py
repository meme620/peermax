"""脚本共享的原子模块。

放置规则：只放无状态、无模块级副作用的纯工具；不放读取 registry 或数据的逻辑。
目录层级：`scripts/<dir>/x.py` 里用 `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))` 后 `from lib.xxx import ...`。
"""
