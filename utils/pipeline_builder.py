"""
Validate, compile and inspect KFP v2 pipelines written by the assistant or attached by a user.

Compiling a pipeline means importing its module and calling the ``@dsl.pipeline``
function, so LLM-written code runs on the app server at that point. Two layers keep
that safe:

1. ``validate_pipeline_source`` checks the AST against an allowlist before anything
   runs. Module top level may only import kfp (plus a few harmless stdlib modules),
   assign literal constants and define ``@dsl.component`` / ``@dsl.pipeline``
   functions. The pipeline body, which runs at compile time, may only call the
   module's components, a few builtins, the ``dsl`` control-flow helpers and task
   methods. Component bodies are not checked: they never run here, only in the
   user's own Kubeflow pods.
2. ``compile_pipeline_source`` compiles in a separate ``python -I`` process with a
   minimal environment (no API keys or tokens) and a timeout.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import resource
except ImportError:  # Windows
    resource = None
from typing import Any, Dict, List, Optional, Tuple

import yaml

MAX_SOURCE_BYTES = 100_000
COMPILE_TIMEOUT_SECONDS = 60
COMPILE_MEMORY_BYTES = 2 * 1024 ** 3

# Modules a pipeline file may import. Only kfp names may be used in the pipeline
# body; the others are allowed so a normal pipeline file (e.g. one that compiles
# itself under `if __name__ == "__main__":`) passes unchanged.
_KFP_MODULES = {"kfp", "kfp.dsl", "kfp.compiler"}
_OTHER_ALLOWED_MODULES = {"__future__", "typing", "pathlib", "collections", "json", "math", "datetime"}

# Functions on `dsl` (or imported from kfp.dsl) the pipeline body may call.
_DSL_CALLS = {
    "If", "Elif", "Else", "Condition", "ParallelFor", "ExitHandler",
    "Collected", "OneOf", "importer",
}
_DECORATORS = {"component", "pipeline"}
# Names that may be imported from kfp.dsl (plus PIPELINE_* placeholder constants).
_DSL_IMPORTABLE = _DSL_CALLS | _DECORATORS | {
    "Input", "Output", "InputPath", "OutputPath", "Artifact", "Dataset", "Model", "Metrics",
    "ClassificationMetrics", "SlicedClassificationMetrics", "HTML", "Markdown",
}
# Methods the pipeline body may call on tasks, outputs and plain containers.
_TASK_METHODS = {
    "after", "set_caching_options", "set_cpu_limit", "set_cpu_request",
    "set_memory_limit", "set_memory_request", "set_env_variable",
    "set_accelerator_type", "set_accelerator_limit", "set_retry",
    "ignore_upstream_failure", "add_node_selector_constraint",
    "items", "keys", "values", "get", "append", "update",
}
# No sorted/min/max/map: their key= would let the body call any method it can reach.
_SAFE_BUILTINS = {
    "dict", "list", "tuple", "range", "len", "str", "int", "float", "bool", "zip", "enumerate",
}
_FORBIDDEN_IN_PIPELINE = (
    ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
    ast.Lambda, ast.Global, ast.Nonlocal, ast.While, ast.Try, ast.Await, ast.Yield,
    ast.YieldFrom, ast.AsyncWith, ast.AsyncFor, ast.Delete,
)
_CONSTANT_NAME = re.compile(r"[A-Z][A-Z0-9_]*")

_COMPILE_RUNNER = """
import importlib.util, sys
from kfp import compiler
source_path, output_path, func_name = sys.argv[1:4]
spec = importlib.util.spec_from_file_location("swarm_user_pipeline", source_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
compiler.Compiler().compile(getattr(module, func_name), output_path)
"""


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def _is_main_guard(node: ast.stmt) -> bool:
    """True for a top-level `if __name__ == "__main__":` block (never runs when compiled here)."""
    test = getattr(node, "test", None)
    return (
        isinstance(node, ast.If)
        and isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )


def _name_root(node: ast.expr) -> Optional[str]:
    """Root name of an attribute/subscript chain such as `a.b[0].c` (None if it starts elsewhere)."""
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


class _Validator:
    def __init__(self, tree: ast.Module):
        self.tree = tree
        self.errors: List[str] = []
        self.constants: set[str] = set()
        self.components: set[str] = set()
        self.dsl_modules: set[str] = set()  # names bound to the kfp.dsl module (`dsl`)
        self.kfp_roots: set[str] = set()  # names bound to the kfp package (`import kfp`)
        self.compiler_names: set[str] = set()  # names bound to kfp.compiler
        self.dsl_imports: Dict[str, str] = {}  # bound name -> original name, from `from kfp.dsl import ...`
        self.other_imports: set[str] = set()
        self.pipelines: List[ast.FunctionDef] = []
        self.bindings: Dict[str, Tuple[str, str]] = {}  # module-level name -> (kind, what it refers to)

    def error(self, node: ast.AST, message: str) -> None:
        self.errors.append(f"line {getattr(node, 'lineno', '?')}: {message}")

    def bind(self, node: ast.AST, name: str, kind: str, target: str) -> bool:
        """Record a module-level name; a second, different binding of the same name is an error.

        The checks below trust what a name was first bound to, so rebinding (e.g. importing
        pathlib.Path as an imported dsl helper's name) must not be possible.
        """
        previous = self.bindings.get(name)
        if previous is not None and previous != (kind, target):
            self.error(node, f"'{name}' is bound more than once at module level")
            return False
        self.bindings[name] = (kind, target)
        return True

    # -- names --------------------------------------------------------------

    def module_names(self) -> set[str]:
        """Names that refer to imported modules or kfp objects and must not be used freely."""
        return self.dsl_modules | self.kfp_roots | self.compiler_names | self.other_imports | set(self.dsl_imports)

    def is_dsl_ref(self, node: ast.expr) -> bool:
        """`dsl` or `kfp.dsl`."""
        if isinstance(node, ast.Name):
            return node.id in self.dsl_modules
        return (
            isinstance(node, ast.Attribute) and node.attr == "dsl"
            and isinstance(node.value, ast.Name) and node.value.id in self.kfp_roots
        )

    def dsl_member(self, node: ast.expr) -> Optional[str]:
        """Return X for `dsl.X` / `kfp.dsl.X` / a name imported from kfp.dsl, else None."""
        if isinstance(node, ast.Attribute) and self.is_dsl_ref(node.value):
            return node.attr
        if isinstance(node, ast.Name) and node.id in self.dsl_imports:
            return self.dsl_imports[node.id]
        return None

    def decorator_kind(self, decorator: ast.expr) -> Optional[str]:
        """'component' or 'pipeline' for exactly dsl.component / dsl.pipeline (optionally called), else None."""
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        member = self.dsl_member(target)
        return member if member in _DECORATORS else None

    # -- top level ---------------------------------------------------------

    def run(self) -> List[str]:
        body = self.tree.body
        checked: List[ast.stmt] = []
        for index, node in enumerate(body):
            if index == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                continue  # module docstring
            if _is_main_guard(node):
                continue
            checked.append(node)
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                self.check_import(node)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                self.check_constant(node)
            elif isinstance(node, ast.FunctionDef):
                self.check_function(node)
            else:
                self.error(node, f"{type(node).__name__} is not allowed at module level; "
                                 "only imports, constants, @dsl.component and @dsl.pipeline functions")

        if not self.pipelines:
            self.errors.append("no @dsl.pipeline function found")
        elif len(self.pipelines) > 1:
            names = ", ".join(p.name for p in self.pipelines)
            self.errors.append(f"exactly one @dsl.pipeline function is allowed (found {names})")
        for pipeline in self.pipelines:
            self.check_pipeline_body(pipeline)
        for node in checked:
            self.scan_outside_bodies(node)
        return self.errors

    def scan_outside_bodies(self, stmt: ast.stmt) -> None:
        """Backstop: outside function bodies only the decorator itself may be a call, and no dunders."""
        allowed_calls = set()
        stack: List[ast.AST] = [stmt]
        while stack:
            node = stack.pop()
            if isinstance(node, ast.FunctionDef):
                allowed_calls.update(d for d in node.decorator_list if isinstance(d, ast.Call))
                stack.extend(node.decorator_list)
                stack.append(node.args)
                if node.returns is not None:
                    stack.append(node.returns)
                continue  # bodies: components never run here; the pipeline body is checked separately
            if isinstance(node, ast.Call) and node not in allowed_calls:
                self.error(node, "function calls are only allowed inside component and pipeline functions")
            elif isinstance(node, (ast.Lambda, ast.NamedExpr)):
                self.error(node, f"{type(node).__name__} is not allowed outside function bodies")
            elif isinstance(node, ast.Name) and _is_dunder(node.id) or \
                    isinstance(node, ast.Attribute) and _is_dunder(node.attr) or \
                    isinstance(node, ast.arg) and _is_dunder(node.arg):
                self.error(node, "dunder names are not allowed")
            stack.extend(ast.iter_child_nodes(node))

    def check_import(self, node: ast.stmt) -> None:
        for alias in node.names:
            if _is_dunder(alias.name.split(".")[-1]) and alias.name != "__future__" or \
                    alias.asname and _is_dunder(alias.asname):
                self.error(node, "dunder names are not allowed in imports")
                return
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                target = alias.name if alias.asname else bound
                if not self.bind(node, bound, "module", target):
                    continue
                if alias.name in _KFP_MODULES:
                    if alias.asname is None:
                        self.kfp_roots.add("kfp")
                    elif alias.name == "kfp.dsl":
                        self.dsl_modules.add(alias.asname)
                    elif alias.name == "kfp.compiler":
                        self.compiler_names.add(alias.asname)
                    else:
                        self.kfp_roots.add(alias.asname)
                elif alias.name in _OTHER_ALLOWED_MODULES:
                    self.other_imports.add((alias.asname or alias.name).split(".")[0])
                else:
                    self.error(node, f"import of '{alias.name}' is not allowed at module level; "
                                     "import libraries inside the component function instead")
            return

        module = node.module or ""
        if node.level or (module not in _KFP_MODULES and module not in _OTHER_ALLOWED_MODULES):
            self.error(node, f"import from '{module}' is not allowed at module level; "
                             "import libraries inside the component function instead")
            return
        for alias in node.names:
            if alias.name == "*":
                self.error(node, "wildcard imports are not allowed")
                continue
            bound = alias.asname or alias.name
            if not self.bind(node, bound, "from", f"{module}.{alias.name}"):
                continue
            if module == "kfp":
                if alias.name == "dsl":
                    self.dsl_modules.add(bound)
                elif alias.name == "compiler":
                    self.compiler_names.add(bound)
                else:
                    self.error(node, f"'from kfp import {alias.name}' is not allowed; use 'from kfp import dsl'")
            elif module == "kfp.dsl":
                if alias.name not in _DSL_IMPORTABLE and not alias.name.startswith("PIPELINE_"):
                    self.error(node, f"'from kfp.dsl import {alias.name}' is not allowed")
                    continue
                self.dsl_imports[bound] = alias.name
            elif module == "kfp.compiler":
                self.compiler_names.add(bound)
            elif module != "__future__":
                self.other_imports.add(bound)

    def is_literal(self, node: Optional[ast.expr]) -> bool:
        if node is None:
            return True
        if isinstance(node, ast.Constant):
            return True
        if isinstance(node, ast.Name):
            # module constants, or placeholders imported from kfp.dsl
            return node.id in self.constants or \
                bool(_CONSTANT_NAME.fullmatch(self.dsl_imports.get(node.id, "")))
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            return all(self.is_literal(e) for e in node.elts)
        if isinstance(node, ast.Dict):
            return all(k is not None and self.is_literal(k) for k in node.keys) and all(
                self.is_literal(v) for v in node.values
            )
        if isinstance(node, ast.BinOp):
            # no * or **: "x" * 10**9 would exhaust memory at import time
            return isinstance(node.op, (ast.Add, ast.Sub, ast.Div, ast.FloorDiv, ast.Mod, ast.BitOr)) and \
                self.is_literal(node.left) and self.is_literal(node.right)
        if isinstance(node, ast.UnaryOp):
            return self.is_literal(node.operand)
        if isinstance(node, ast.JoinedStr):
            return all(self.is_literal(v) for v in node.values)
        if isinstance(node, ast.FormattedValue):
            return self.is_literal(node.value) and self.is_literal(node.format_spec)
        if isinstance(node, ast.Attribute):
            # dsl.PIPELINE_JOB_ID_PLACEHOLDER and similar placeholders
            return self.is_dsl_ref(node.value) and bool(_CONSTANT_NAME.fullmatch(node.attr))
        return False

    def check_constant(self, node: ast.stmt) -> None:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if isinstance(node, ast.AnnAssign):
            self.check_annotation(node.annotation)
        if node.value is None or not self.is_literal(node.value):
            self.error(node, "module-level assignments must be literal constants")
            return
        for target in targets:
            if not isinstance(target, ast.Name) or _is_dunder(target.id):
                self.error(node, "module-level assignments must bind plain names")
                return
            if self.bind(node, target.id, "constant", str(id(node))):
                self.constants.add(target.id)

    def check_annotation(self, node: Optional[ast.expr], depth: int = 0) -> None:
        if node is None:
            return
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                # String annotations can be evaluated later (typing.get_type_hints), so check them as code.
                try:
                    parsed = ast.parse(sub.value.strip(), mode="eval").body
                except SyntaxError:
                    self.error(sub, "string type annotations must be valid type expressions")
                    return
                if depth >= 2:
                    self.error(sub, "nested string type annotations are not allowed")
                    return
                for inner in ast.walk(parsed):  # report errors at the annotation's own line
                    inner.lineno = sub.lineno
                self.check_annotation(parsed, depth + 1)
                continue
            if not isinstance(sub, (ast.Name, ast.Attribute, ast.Subscript, ast.Tuple, ast.List,
                                    ast.Constant, ast.BinOp, ast.BitOr, ast.Load)):
                self.error(sub, "type annotations may only name types "
                                "(use Output[...] artifacts instead of NamedTuple)")
                return
            if isinstance(sub, ast.Name) and _is_dunder(sub.id) or \
                    isinstance(sub, ast.Attribute) and _is_dunder(sub.attr):
                self.error(sub, "dunder names are not allowed")
                return

    def check_function(self, node: ast.FunctionDef) -> None:
        kinds = [self.decorator_kind(d) for d in node.decorator_list]
        if len(node.decorator_list) != 1 or kinds[0] is None:
            self.error(node, f"function '{node.name}' must have exactly one decorator, kfp's "
                             "@dsl.component or @dsl.pipeline; put helpers inside the component")
            return
        decorator = node.decorator_list[0]
        if isinstance(decorator, ast.Call):
            for arg in [*decorator.args, *(k.value for k in decorator.keywords)]:
                if not self.is_literal(arg):
                    self.error(decorator, f"arguments of the decorator on '{node.name}' must be "
                                          "literals or module-level constants")
                    break
        args = node.args
        for default in [*args.defaults, *(d for d in args.kw_defaults if d is not None)]:
            if not self.is_literal(default):
                self.error(default, f"default values of '{node.name}' must be literals or module-level constants")
        for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
            self.check_annotation(arg.annotation)
        if args.vararg or args.kwarg:
            self.error(node, f"'{node.name}' may not take *args or **kwargs")
        self.check_annotation(node.returns)

        if not self.bind(node, node.name, kinds[0], str(id(node))):
            return
        if kinds[0] == "component":
            self.components.add(node.name)
        else:
            self.pipelines.append(node)

    # -- pipeline body (runs at compile time) -------------------------------

    def check_pipeline_body(self, pipeline: ast.FunctionDef) -> None:
        params = {a.arg for a in [*pipeline.args.posonlyargs, *pipeline.args.args, *pipeline.args.kwonlyargs]}
        protected = self.module_names() | self.components | self.constants | _SAFE_BUILTINS
        for name in sorted(params & protected):
            self.error(pipeline, f"pipeline parameter '{name}' shadows a component, builtin, constant or import")
        restricted = self.module_names() - params
        for stmt in pipeline.body:
            parents = {child: parent for parent in ast.walk(stmt) for child in ast.iter_child_nodes(parent)}
            for node in ast.walk(stmt):
                if isinstance(node, _FORBIDDEN_IN_PIPELINE):
                    self.error(node, f"{type(node).__name__} is not allowed in the pipeline function")
                elif isinstance(node, ast.Name):
                    if isinstance(node.ctx, (ast.Store, ast.Del)) and node.id in protected:
                        self.error(node, f"'{node.id}' cannot be reassigned in the pipeline function")
                    elif node.id in self.components and not (
                        isinstance(parents.get(node), ast.Call) and parents[node].func is node
                    ):
                        self.error(node, f"component '{node.id}' may only be called in the pipeline function")
                    else:
                        self.check_pipeline_name(node, parents.get(node), restricted)
                elif isinstance(node, ast.Attribute):
                    self.check_pipeline_attribute(node, parents.get(node), restricted)
                elif isinstance(node, ast.Call):
                    self.check_pipeline_call(node, restricted)

    def check_pipeline_name(self, node: ast.Name, parent: Optional[ast.AST], restricted: set[str]) -> None:
        if _is_dunder(node.id):
            self.error(node, "dunder names are not allowed")
            return
        if node.id not in restricted:
            return
        # A module name may only appear as the base of an allowed dsl.X / kfp.dsl.X access.
        if isinstance(parent, ast.Attribute) and parent.value is node and \
                (node.id in self.dsl_modules or node.id in self.kfp_roots):
            return
        original = self.dsl_imports.get(node.id)
        if original is not None and (
            original in _DSL_CALLS and isinstance(parent, ast.Call) and parent.func is node
            or _CONSTANT_NAME.fullmatch(original)
        ):
            return
        self.error(node, f"'{node.id}' cannot be used here in the pipeline function; "
                         "only components, parameters, dsl.<helper>(...) and dsl.<CONSTANT>")

    def check_pipeline_attribute(self, node: ast.Attribute, parent: Optional[ast.AST], restricted: set[str]) -> None:
        if _is_dunder(node.attr):
            self.error(node, "dunder attributes are not allowed")
            return
        if node.attr == "set_display_name":
            self.error(node, "set_display_name() is not allowed: the cluster's KFP driver "
                             "crashes when a step's display name differs from its task name")
            return
        if node.attr.startswith("_"):
            self.error(node, "private attributes are not allowed in the pipeline function")
            return
        if _name_root(node) not in restricted:
            return  # attributes of tasks, parameters and local values
        if self.is_dsl_ref(node) and isinstance(parent, ast.Attribute) and parent.value is node:
            return  # the `kfp.dsl` part of `kfp.dsl.X`
        if self.is_dsl_ref(node.value) and (node.attr in _DSL_CALLS or _CONSTANT_NAME.fullmatch(node.attr)):
            return
        self.error(node, f"'{ast.unparse(node)}' cannot be used in the pipeline function; "
                         "only dsl.<helper>(...) and dsl.<CONSTANT>")

    def check_pipeline_call(self, node: ast.Call, restricted: set[str]) -> None:
        func = node.func
        if isinstance(func, ast.Name):
            original = self.dsl_imports.get(func.id)
            if func.id in self.components or (func.id in _SAFE_BUILTINS and func.id not in restricted) or \
                    original in _DSL_CALLS:
                return
            self.error(node, f"calling '{func.id}' is not allowed in the pipeline function; "
                             "only components defined in this file, dsl helpers and basic builtins")
            return
        if isinstance(func, ast.Attribute):
            if _name_root(func) in restricted:
                if self.is_dsl_ref(func.value) and func.attr in _DSL_CALLS:
                    return
                self.error(node, f"calling '{ast.unparse(func)}' is not allowed in the pipeline function")
                return
            if func.attr in _TASK_METHODS or func.attr == "set_display_name":  # reported by the Attribute check
                return
            self.error(node, f"method '{func.attr}' is not allowed in the pipeline function")
            return
        self.error(node, "only direct function and method calls are allowed in the pipeline function")


def _validate(source: str) -> Tuple[List[str], Optional[str]]:
    if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        return [f"source is larger than {MAX_SOURCE_BYTES // 1000} KB"], None
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"syntax error at line {exc.lineno}: {exc.msg}"], None
    validator = _Validator(tree)
    errors = validator.run()
    return errors, validator.pipelines[0].name if validator.pipelines else None


def validate_pipeline_source(source: str) -> List[str]:
    """Return a list of problems with a pipeline source file (empty when it is safe to compile)."""
    return _validate(source)[0]


def compile_pipeline_source(
    source: str, timeout: int = COMPILE_TIMEOUT_SECONDS
) -> Tuple[Optional[str], Optional[str]]:
    """Validate and compile pipeline source in an isolated subprocess.

    Returns:
        (yaml_text, None) on success, or (None, error message) otherwise.
    """
    problems, func_name = _validate(source)
    if problems:
        return None, "Pipeline source rejected:\n- " + "\n- ".join(problems)

    with tempfile.TemporaryDirectory(prefix="swarm-pipeline-") as workdir:
        source_path = Path(workdir) / "pipeline_source.py"
        output_path = Path(workdir) / "pipeline.yaml"
        source_path.write_text(source, encoding="utf-8")
        env = {"PATH": os.environ.get("PATH", ""), "HOME": workdir, "LANG": "C.UTF-8"}
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-W", "ignore", "-c", _COMPILE_RUNNER, str(source_path), str(output_path), func_name],
                cwd=workdir,
                env=env,
                preexec_fn=(lambda: _limit_compile_resources(timeout)) if resource else None,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return None, f"Compilation timed out after {timeout} seconds"
        if result.returncode != 0 or not output_path.is_file():
            return None, "Compilation failed:\n" + _trim_traceback(result.stderr)
        return output_path.read_text(encoding="utf-8"), None


def _limit_compile_resources(timeout: int) -> None:
    """Cap CPU time and memory of the compile subprocess (runs in the child before exec)."""
    for limit, value in ((resource.RLIMIT_CPU, timeout + 5), (resource.RLIMIT_AS, COMPILE_MEMORY_BYTES)):
        try:
            resource.setrlimit(limit, (value, value))
        except (ValueError, OSError):
            pass  # macOS does not enforce RLIMIT_AS; the wall-clock timeout still applies


def _trim_traceback(stderr: str, max_lines: int = 15) -> str:
    """Keep the frames from the user's file and the final error, dropping kfp-internal frames."""
    lines = stderr.strip().splitlines()
    start = max((i for i, line in enumerate(lines) if line.startswith("Traceback")), default=0)
    lines = lines[start:]
    kept: List[str] = []
    for index, line in enumerate(lines):
        if "pipeline_source.py" in line:
            kept.append(re.sub(r'File ".*?pipeline_source.py"', 'File "pipeline.py"', line.strip()))
            if index + 1 < len(lines):
                kept.append("  " + lines[index + 1].strip())
    tail = [line for line in lines if line and not line.startswith((" ", "Traceback"))]
    kept.extend(tail[-3:])
    return "\n".join(kept[-max_lines:]) or "no error output"


# -- compiled spec ------------------------------------------------------------

def unwrap_pipeline_spec(spec: Any) -> Any:
    """Specs uploaded with a platform spec come back as {"pipeline_spec": ..., "platform_spec": ...}."""
    if isinstance(spec, dict) and "root" not in spec and isinstance(spec.get("pipeline_spec"), dict):
        return spec["pipeline_spec"]
    return spec


def pipeline_root_parameters(spec: Any) -> Dict[str, Dict[str, Any]]:
    """Root input parameter definitions of a pipeline spec ({} if there are none)."""
    spec = unwrap_pipeline_spec(spec)
    try:
        parameters = spec["root"]["inputDefinitions"]["parameters"]
    except (KeyError, TypeError):
        return {}
    return parameters if isinstance(parameters, dict) else {}


def load_pipeline_yaml(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Parse compiled KFP v2 pipeline YAML (optionally followed by a platform spec document).

    Returns:
        (pipeline spec dict, None) or (None, error message).
    """
    try:
        documents = [d for d in yaml.safe_load_all(text) if d is not None]
    except yaml.YAMLError as exc:
        return None, f"Not valid YAML: {exc}"
    if not documents or not isinstance(documents[0], dict):
        return None, "The file does not contain a pipeline spec"
    spec = unwrap_pipeline_spec(documents[0])
    if spec.get("kind") == "Workflow" or "apiVersion" in spec:
        return None, "This looks like a KFP v1 / Argo workflow; only KFP v2 pipeline specs are supported"
    missing = [key for key in ("pipelineInfo", "root", "deploymentSpec") if key not in spec]
    if missing:
        return None, f"Not a KFP v2 pipeline spec (missing {', '.join(missing)})"
    return spec, None


_PACKAGE_TOKEN = re.compile(r"'([A-Za-z0-9_.\-]+(?:\[[^\]]*\])?\s*(?:==|>=|<=|~=|!=|<|>)[^'\s]+)'")


def inspect_pipeline_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize a compiled pipeline spec for the user: name, inputs, steps, images, packages."""
    spec = unwrap_pipeline_spec(spec)
    info = spec.get("pipelineInfo") or {}
    inputs = {
        name: {
            "type": definition.get("parameterType"),
            "default": definition.get("defaultValue"),
            "optional": bool(definition.get("isOptional")),
        }
        for name, definition in pipeline_root_parameters(spec).items()
    }
    tasks = list(((spec.get("root") or {}).get("dag") or {}).get("tasks") or {})
    images, packages = set(), set()
    for executor in ((spec.get("deploymentSpec") or {}).get("executors") or {}).values():
        container = executor.get("container") or {}
        if container.get("image"):
            images.add(container["image"])
        command = " ".join(str(part) for part in container.get("command") or [])
        for package in _PACKAGE_TOKEN.findall(command):
            if not package.startswith(("kfp", "typing-extensions")):
                packages.add(package)
    return {
        "name": info.get("name"),
        "description": info.get("description"),
        "inputs": inputs,
        "steps": tasks,
        "images": sorted(images),
        "packages": sorted(packages),
    }
