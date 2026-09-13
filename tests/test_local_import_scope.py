"""静态守卫：**函数里的局部导入，不能拿到别的函数里用**。

## 为什么要有这个测试

2026-09-13 用户点向导里的「列出模型」直接报错：

```
✗ NameError: name 'probe_endpoint' is not defined
```

根因：

```python
def _list_models(self):
    from app.translate.openai_compat import probe_endpoint   # 只进这个函数的局部作用域
    t = _CheckThread(lambda: self._do_list(base))            # 真正干活的是另一个方法

def _do_list(self, base):
    ok, msg, models = probe_endpoint(...)                    # ← 这里根本没定义
```

`probe_endpoint` 是模块级函数，所以"在 A 里导入、在 B 里使用"这种写法在
**同一个模块内**不会报 ImportError，只在**运行时踩到那条分支**才炸——
而这条分支（点按钮）恰恰是单元测试最容易漏掉的路径。

这个测试用 AST 把这种写法全仓库扫一遍：**局部导入的名字，若没有模块级同名定义，
却出现在另一个函数的"读取"位置，就判定为 bug**。
判定刻意保守（只在"确定没定义"时报），宁可漏报不可误报。
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class _ModuleBinder(ast.NodeVisitor):
    """只收集**模块级**绑定的名字。

    关键：**不进函数体**。函数里的局部导入绝不能算成"模块里有定义"——
    否则这个检查器就永远不报警（第一版就踩了这个坑，自证用例当场抓到）。
    类体也不进（保守：类属性本来就不该当全局名用）。
    """

    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802 - ast 命名
        self.names.add(node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self.names.add(node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self.names.add(node.name)

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            self.names.add(alias.asname or alias.name.split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        for alias in node.names:
            if alias.name != "*":
                self.names.add(alias.asname or alias.name)

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        if isinstance(node.ctx, ast.Store):
            self.names.add(node.id)

    def visit_arg(self, node: ast.arg) -> None:  # noqa: N802
        self.names.add(node.arg)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:  # noqa: N802
        if node.name:
            self.names.add(node.name)
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:  # noqa: N802
        self.names.update(node.names)

    visit_Nonlocal = visit_Global


def _module_bound_names(tree: ast.Module) -> set[str]:
    binder = _ModuleBinder()
    for node in tree.body:
        binder.visit(node)
    return binder.names


def _ancestors(tree: ast.Module) -> dict[int, list[int]]:
    """每个函数节点 id → 外层函数**节点 id** 列表（用来识别合法的闭包）。"""
    out: dict[int, list[int]] = {}

    def walk(node: ast.AST, stack: list[int]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out[id(child)] = list(stack)
                walk(child, stack + [id(child)])
            else:
                walk(child, stack)

    walk(tree, [])
    return out


def find_cross_function_local_imports(source: str) -> list[tuple[str, str, str]]:
    """返回 ``[(导入所在的函数, 使用它的函数, 名字), …]``。"""
    tree = ast.parse(source)
    module_names = _module_bound_names(tree)
    parents = _ancestors(tree)

    funcs = [
        (n.name, n)
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    # id(节点) → 名字，用来判断"使用者在不在导入者的内部"（闭包是合法的）
    id_to_name = {id(n): n.name for _name, n in funcs}

    imports: dict[str, set[str]] = {}
    used: dict[str, set[str]] = {}
    bound: dict[str, set[str]] = {}
    nesting: dict[str, list[str]] = {}

    for name, node in funcs:
        local_imports: set[str] = set()
        loads: set[str] = set()
        binds: set[str] = set()
        for sub in ast.walk(node):
            if isinstance(sub, ast.Import):
                for a in sub.names:
                    local_imports.add(a.asname or a.name.split(".")[0])
            elif isinstance(sub, ast.ImportFrom):
                for a in sub.names:
                    if a.name != "*":
                        local_imports.add(a.asname or a.name)
            elif isinstance(sub, ast.Name):
                (loads if isinstance(sub.ctx, ast.Load) else binds).add(sub.id)
            elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub is not node:
                binds.add(sub.name)
            elif isinstance(sub, ast.ClassDef) and sub is not node:
                binds.add(sub.name)
            elif isinstance(sub, ast.arg):
                binds.add(sub.arg)
            elif isinstance(sub, ast.ExceptHandler) and sub.name:
                binds.add(sub.name)
            elif isinstance(sub, (ast.Global, ast.Nonlocal)):
                binds.update(sub.names)
        imports[name] = local_imports
        used[name] = loads
        # 局部导入的名字在本函数内也算"已绑定"（同一函数里导入+使用是正常写法）
        bound[name] = binds | local_imports
        nesting[name] = [id_to_name.get(pid, "?") for pid in parents.get(id(node), [])]

    problems: list[tuple[str, str, str]] = []
    for owner, names in imports.items():
        for candidate in names:
            if candidate in module_names:
                continue
            for user, loads in used.items():
                if user == owner:
                    continue
                # 内层函数用外层函数的局部导入 = 闭包，合法
                if owner in nesting.get(user, []):
                    continue
                if candidate in loads and candidate not in bound[user]:
                    problems.append((owner, user, candidate))
    return problems


# --------------------------------------------------------------------------- #
# 先自证：这个检查器真的能抓到那种写法（不然它就是一个永远不报警的摆设）
# --------------------------------------------------------------------------- #
BAD_SOURCE = '''
def _list_models(self):
    from app.translate.openai_compat import probe_endpoint
    start(self._do_list)

def _do_list(self, base):
    ok, msg, models = probe_endpoint(base)
    return msg
'''

GOOD_SAME_FUNCTION = '''
def only_one(self):
    from app.translate.openai_compat import probe_endpoint
    return probe_endpoint("x")
'''

GOOD_MODULE_LEVEL = '''
from app.translate.openai_compat import probe_endpoint

def a(self):
    return 1

def b(self):
    return probe_endpoint("x")
'''


def test_checker_catches_the_real_bug():
    found = find_cross_function_local_imports(BAD_SOURCE)
    assert found == [("_list_models", "_do_list", "probe_endpoint")]


def test_checker_accepts_same_function_usage():
    assert find_cross_function_local_imports(GOOD_SAME_FUNCTION) == []


def test_checker_accepts_module_level_import():
    assert find_cross_function_local_imports(GOOD_MODULE_LEVEL) == []


# --------------------------------------------------------------------------- #
# 全仓库扫一遍
# --------------------------------------------------------------------------- #
def test_no_cross_function_local_imports_in_repo():
    files = [ROOT / "main.py", ROOT / "frontend.py"]
    for folder in ("app", "scripts", "tests"):
        files += sorted((ROOT / folder).rglob("*.py"))

    problems: list[str] = []
    for path in files:
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        try:
            found = find_cross_function_local_imports(source)
        except SyntaxError as exc:  # 语法错误另有人管，这里只做这一件事
            problems.append(f"{path.relative_to(ROOT)}: 语法错误 {exc}")
            continue
        for owner, user, name in found:
            problems.append(
                f"{path.relative_to(ROOT)}: {owner}() 里局部导入 {name}，"
                f"却在 {user}() 里使用"
            )

    assert not problems, "存在「跨函数使用局部导入」的写法（点开那条分支就会 NameError）：\n" + "\n".join(
        problems
    )
