"""
function_graphs.py
-------------------
Turns a typed-out math expression like "sin(sqrt(x^2 + y^2) * 4)" into a
safe, vectorized callable(u, v) -> height field, so AR mode's 3D Surface
can be built from a function someone wrote instead of only the four
built-in presets in ar3d.py (SURFACE_FUNCS).

Two pieces, each worth understanding on its own:

1. SAFETY. The expression comes from a text file a person edited (see
   ar_overlay.FUNC_FOLDER / list_function_files below) - never run it
   through eval() directly, since a string could just as easily be
   "__import__('os').system('...')" as it could be a function graph.
   Instead the expression is parsed into a Python AST (ast.parse) and
   walked node-by-node BEFORE anything is executed: only arithmetic,
   comparisons, and calls to a fixed whitelist of math functions are
   allowed. Anything else (attribute access, subscripting, name lookups
   outside the whitelist, string/byte literals, comprehensions,
   lambdas, walrus, you name it) is rejected with FunctionParseError
   before a single line of the expression ever runs. Only once the walk
   passes does compile()+eval() run, and even then with an empty
   __builtins__ and a namespace containing only the whitelisted names -
   two independent layers, not one.

2. VECTORIZATION. Surface3D evaluates the field over a whole N x N grid
   at once (u, v are numpy arrays, not single numbers) purely for
   speed - evaluating one Python float at a time for a 21x21 grid would
   be ~400 separate calls every time the surface needs rebuilding. The
   whitelist below maps names like "sin" to numpy's ufuncs (np.sin, not
   python's math.sin) specifically so a typed expression "just works"
   elementwise across the grid with no special-casing from the caller.
"""
import ast
import math
import numpy as np

__all__ = ["FunctionParseError", "compile_function", "list_function_files",
           "ensure_sample_functions", "FUNC_FOLDER"]


class FunctionParseError(ValueError):
    """Raised for anything from a syntax error to a disallowed name/
    operation - str(e) is always meant to be shown to the person who
    typed the expression, not just logged."""
    pass


# Every name an expression is allowed to use. Deliberately numpy
# ufuncs, not the `math` module - see module docstring, part 2.
_ALLOWED_FUNCS = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan, "atan2": np.arctan2,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "exp": np.exp, "log": np.log, "log2": np.log2, "log10": np.log10,
    "sqrt": np.sqrt, "abs": np.abs, "sign": np.sign,
    "hypot": np.hypot, "min": np.minimum, "max": np.maximum,
    "floor": np.floor, "ceil": np.ceil, "round": np.round,
    "clip": np.clip,
}
_ALLOWED_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau}
# Both spellings accepted (x,y reads naturally; u,v matches the grid
# the built-in presets already use) - both just alias the same arrays.
_VAR_ALIASES = {"x": "u", "y": "v", "u": "u", "v": "v"}

_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv, ast.USub, ast.UAdd)
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.IfExp, ast.Compare, ast.BoolOp, ast.And, ast.Or,
    ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq,
) + _ALLOWED_BINOPS


def _validate(node, src):
    if not isinstance(node, _ALLOWED_NODES):
        raise FunctionParseError(
            f"\"{type(node).__name__}\" isn't allowed in a function graph "
            f"(only arithmetic, comparisons, and the listed math functions "
            f"are) - near: {ast.get_source_segment(src, node) or src!r}")
    if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
        raise FunctionParseError(f"only numbers are allowed, not {node.value!r}")
    if isinstance(node, ast.Name) and node.id not in _VAR_ALIASES and node.id not in _ALLOWED_CONSTS:
        raise FunctionParseError(
            f"unknown name \"{node.id}\" - variables are x/y (or u/v), "
            f"constants are pi/e/tau")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
            bad = node.func.id if isinstance(node.func, ast.Name) else "(...)"
            raise FunctionParseError(
                f"unknown function \"{bad}\" - allowed: {', '.join(sorted(_ALLOWED_FUNCS))}")
        if node.keywords:
            raise FunctionParseError(f"\"{node.func.id}\" doesn't take keyword arguments")
        # node.func (the "sin" in "sin(x)") is a Name, but it names a
        # FUNCTION, not a variable - it's already been checked against
        # _ALLOWED_FUNCS just above, so skip the generic Name check
        # below for it (which only knows about variables/constants) and
        # only recurse into the actual arguments.
        for arg in node.args:
            _validate(arg, src)
        return
    for child in ast.iter_child_nodes(node):
        _validate(child, src)


class _Vectorize(ast.NodeTransformer):
    """Rewrites the two constructs that parse and validate fine but
    silently break once x/y are numpy ARRAYS instead of single numbers:

    - `a if cond else b` (ast.IfExp) - Python's own ternary operator
      requires one single True/False answer to pick a branch, but
      `cond` here is a whole array of them (different per grid point),
      so plain eval() raises "truth value of an array is ambiguous".
      Rewritten to a call to _vec_where (np.where), which picks
      elementwise instead of needing one global answer.
    - `a and b` / `a or b` (ast.BoolOp) - same problem, same fix, via
      _vec_and / _vec_or (np.logical_and / np.logical_or).

    This runs AFTER _validate has already confirmed the expression only
    uses the whitelist - it changes HOW an already-approved expression
    is evaluated, not what's allowed in it. The three replacement names
    are added as extra, non-whitelisted entries in compile_function's
    eval namespace (a person's typed expression can't reference
    _vec_where itself directly - it was already rejected as an unknown
    name by _validate before this pass ever runs).
    """

    def visit_IfExp(self, node):
        self.generic_visit(node)
        call = ast.Call(func=ast.Name(id="_vec_where", ctx=ast.Load()),
                        args=[node.test, node.body, node.orelse], keywords=[])
        return ast.copy_location(call, node)

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        fn = "_vec_and" if isinstance(node.op, ast.And) else "_vec_or"
        # a and b and c -> _vec_and(_vec_and(a, b), c) - BoolOp.values can
        # have more than two operands for a chained and/or.
        expr = node.values[0]
        for v in node.values[1:]:
            expr = ast.Call(func=ast.Name(id=fn, ctx=ast.Load()), args=[expr, v], keywords=[])
        return ast.copy_location(expr, node)


def compile_function(expr_text):
    """Parses and validates `expr_text`, returning a callable
    f(u, v) -> numpy array of the same shape as u/v. Raises
    FunctionParseError (with a message safe to show the person who
    wrote the expression) if it's invalid or uses anything not on the
    whitelist. Does NOT evaluate the expression itself - that happens
    once per call to the returned callable, so a bad RUNTIME value
    (e.g. log(negative)) surfaces as NaN/inf in the output rather than
    raising, see _sanitize below.

    Accepts an optional "z = " / "f(x,y) = " style prefix, stripped
    before parsing, since that's how most people will naturally write
    one down even though only the right-hand side is evaluated.
    Accepts "^" for power (as well as Python's own "**"), since that's
    the more common way to write it by hand.
    """
    text = expr_text.strip()
    if "=" in text:
        # Keep only what's after the LAST "=" - handles "z = ...",
        # "f(x, y) = ...", and plain "..." (no "=" at all) uniformly.
        text = text.rsplit("=", 1)[1].strip()
    if not text:
        raise FunctionParseError("empty expression")
    text = text.replace("^", "**")

    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as e:
        raise FunctionParseError(f"couldn't parse \"{text}\": {e.msg}") from e
    _validate(tree, text)              # whitelist check runs on the ORIGINAL ast
    tree = ast.fix_missing_locations(_Vectorize().visit(tree))  # THEN rewritten to vectorize

    code = compile(tree, filename="<function_graph>", mode="eval")
    namespace = dict(_ALLOWED_FUNCS)
    namespace.update(_ALLOWED_CONSTS)
    # Not part of _ALLOWED_FUNCS/exposed to a typed expression directly
    # (an expression calling "_vec_where(...)" itself would already have
    # been rejected by _validate above, before this namespace is even
    # built) - only reachable through the if/and/or rewriting above.
    namespace["_vec_where"] = np.where
    namespace["_vec_and"] = np.logical_and
    namespace["_vec_or"] = np.logical_or

    def f(u, v):
        ns = dict(namespace)
        ns["x"] = ns["u"] = u
        ns["y"] = ns["v"] = v
        try:
            # Division/log/etc. by zero over part of the grid is an
            # expected, everyday thing for a hand-typed expression (try
            # graphing 1/x) - numpy already turns that into inf/nan
            # rather than raising, which _sanitize cleans up below, so
            # its warning is suppressed here rather than printed to the
            # console on every single frame the surface is visible.
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                result = eval(code, {"__builtins__": {}}, ns)  # noqa: S307 - validated above
        except ZeroDivisionError:
            result = np.zeros_like(u)
        return _sanitize(np.asarray(result, dtype=float) + np.zeros_like(u))

    f.source = expr_text.strip()
    return f


def _sanitize(h):
    """Runtime-invalid results (divide-by-zero -> inf, log of a
    negative number -> NaN, etc.) are replaced with 0 rather than left
    to propagate into NaN triangles in the rendered mesh or crash the
    normalization step (h.max()/h.min()) in Surface3D._make_field."""
    return np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)


# -- discovering / seeding the ar_functions/ folder -------------------------
import os  # noqa: E402 (grouped here, next to the one thing that needs it)

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
FUNC_FOLDER = os.path.join(_MODULE_DIR, "ar_functions")

_STARTER_FUNCTIONS = [
    ("gaussian_bump.txt", "z = exp(-(x^2 + y^2) * 3)"),
    ("ripple.txt", "z = sin(sqrt(x^2 + y^2) * 6)"),
    ("saddle.txt", "z = x^2 - y^2"),
    ("interference.txt", "z = sin(6*x) + cos(6*y)"),
    ("twin_peaks.txt", "z = exp(-((x-0.4)^2 + (y-0.3)^2) * 8) + "
                        "0.8*exp(-((x+0.4)^2 + (y+0.2)^2) * 8)"),
    # A height field is one z per (x, y) - it can only ever make
    # dome/bump/saddle shapes, never a silhouette with a concave notch
    # like a heart's top (that needs an implicit curve, a different
    # kind of shape entirely - see ar_functions/README.txt). This is
    # the closest honest equivalent: a smooth heart-shaped MOUND, built
    # from the classic 2D heart curve x^2 + (1.3y - |x|^0.67)^2 = 1,
    # using "how far inside the curve this point is" as the height -
    # no comparisons needed, naturally smooth and domed. (Capped at 6
    # starter files total to match _scan_functions'/_scan_files' shared
    # tile-row budget - a flat-topped if/else variant of this is covered
    # by test_function_graphs.py's own tests instead of shipping as a
    # 7th file that would silently push something else off the row.)
    ("heart_mound.txt", "z = max(0, 1.1 - (x^2 + (1.3*y - abs(x)^0.67)^2))"),
]


def list_function_files(folder=None):
    """.txt files in `folder` (default FUNC_FOLDER), sorted by name -
    same shape/behaviour as ar_overlay.list_csv_files."""
    folder = folder or FUNC_FOLDER
    try:
        names = os.listdir(folder)
    except OSError:
        return []
    return sorted(
        os.path.join(folder, n) for n in names
        if n.lower().endswith(".txt") and not n.startswith((".", "~$"))
        and n.lower() != "readme.txt"          # the folder's own instructions file, not a function
        and os.path.isfile(os.path.join(folder, n))
    )


_README_TEXT = """\
Drop .txt files in this folder and they show up under the \u0192(x) button in AR mode.
(Re-scanned every time you open the Functions row - no restart needed.)

FORMAT: one line, a function of x and y (u/v also work, same thing).
Everything up to and including the last "=" is ignored, so any of these
are fine:

    z = sin(sqrt(x^2 + y^2) * 6)
    f(x, y) = sin(sqrt(x^2 + y^2) * 6)
    sin(sqrt(x^2 + y^2) * 6)

"^" means power (x^2, not XOR) - the usual way people write it by hand.
The result is automatically rescaled to fit the hologram, so it doesn't
matter if your function's output is tiny or huge - only its SHAPE
matters.

ALLOWED: + - * / ^ %, comparisons (<, >, ==, ...), and/or, an if/else
ternary, and: sin cos tan asin acos atan atan2 sinh cosh tanh exp log
log2 log10 sqrt abs sign hypot min max floor ceil round clip, and the
constants pi, e, tau. Nothing else is allowed - no other names, no
imports, no attribute access - an expression using anything else is
refused with an explanation rather than silently doing something
unexpected.

A value that would be undefined at some point on the grid (1/x at x=0,
log of a negative number, etc.) just shows as 0 there rather than
crashing or leaving a hole in the surface.

If a .txt file doesn't parse, it shows up in the Functions row as a red
tile with a "!" instead of being skipped silently - fix the file and
reopen the row (close/open the \u0192(x) button) to pick it up.

One real limitation worth knowing: this is a HEIGHT FIELD (one z per
x,y point), so it can only ever make dome/bump/ripple/saddle shapes -
never a shape with a silhouette that curves back on itself, like a
heart's top notch or a star's points meeting. That needs an "implicit
curve" (a 2D region defined by an inequality, like the classic heart
curve x^2 + (1.3y - |x|^0.67)^2 < 1), not a height function. See
heart_mound.txt for the honest equivalent: a smooth dome over that
curve's interior, not a heart-shaped silhouette.
"""


def ensure_sample_functions():
    """Writes a handful of example .txt files (plus this folder's own
    README) into FUNC_FOLDER so the Functions palette button has
    something to show immediately. Only meant to be called on a true
    first run (mirrors ar_overlay._ensure_sample_csv's caller contract)
    - never overwrites an existing file, so a person's own edits/
    additions (including to the README) are always safe."""
    try:
        os.makedirs(FUNC_FOLDER, exist_ok=True)
        readme_path = os.path.join(FUNC_FOLDER, "README.txt")
        if not os.path.exists(readme_path):
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(_README_TEXT)
        for name, expr in _STARTER_FUNCTIONS:
            path = os.path.join(FUNC_FOLDER, name)
            if os.path.exists(path):
                continue
            with open(path, "w", encoding="utf-8") as f:
                f.write(expr + "\n")
    except OSError:
        pass  # non-fatal - Functions browser will just show "no files found"


def read_function_file(path):
    """Reads one .txt file and compiles it. Raises FunctionParseError
    (file-not-found/unreadable is re-raised as one too, so callers only
    need to catch the one exception type) with a message safe to show
    on a tile/tooltip."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        raise FunctionParseError(f"couldn't read {os.path.basename(path)}: {e}") from e
    return compile_function(text)
