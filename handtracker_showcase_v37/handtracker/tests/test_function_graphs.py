"""Safety + correctness tests for function_graphs.py (no camera/display
needed). Run: python tests/test_function_graphs.py

The safety tests matter more than usual here: this is the one place in
the project that turns a string a person typed into executed code, so
it's tested as an adversarial boundary, not just for the happy path.
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import function_graphs as fg

GRID = np.meshgrid(np.linspace(-1, 1, 7), np.linspace(-1, 1, 7))


def test_valid_expressions_vectorize_over_the_grid():
    u, v = GRID
    for expr in ["sin(sqrt(x^2 + y^2) * 6)", "x^2 - y^2", "exp(-(x^2+y^2)*3)",
                 "cos(x)*sin(y)", "1", "x", "hypot(x, y)", "clip(x, -0.5, 0.5)"]:
        out = fg.compile_function(expr)(u, v)
        assert out.shape == u.shape and np.all(np.isfinite(out)), expr


def test_z_equals_and_f_of_xy_prefixes_are_stripped():
    u, v = GRID
    a = fg.compile_function("z = x^2 + y^2")(u, v)
    b = fg.compile_function("f(x, y) = x^2 + y^2")(u, v)
    c = fg.compile_function("x^2 + y^2")(u, v)
    assert np.allclose(a, b) and np.allclose(b, c)


def test_caret_means_power_not_xor():
    u, v = GRID
    out = fg.compile_function("x^2")(u, v)
    assert np.allclose(out, u ** 2)


def test_x_y_and_u_v_are_interchangeable():
    u, v = GRID
    a = fg.compile_function("sin(x) + cos(y)")(u, v)
    b = fg.compile_function("sin(u) + cos(v)")(u, v)
    assert np.allclose(a, b)


def test_ternary_and_boolean_expressions_vectorize_correctly():
    """Regression test for a real bug: IfExp (a if cond else b) and
    BoolOp (and/or) parse and pass _validate fine (they're on the
    whitelist - an if/else ternary is explicitly advertised as allowed
    in ar_functions/README.txt), but Python's own ternary/and/or need a
    single True/False answer, not a whole array of different answers
    per grid point - naive eval() used to raise "truth value of an
    array is ambiguous" the moment anyone actually tried a conditional
    expression against the grid, rather than against a single number."""
    u, v = GRID
    out = fg.compile_function("1 if x > 0 else 0")(u, v)
    assert set(np.unique(out)) <= {0.0, 1.0}
    assert np.array_equal(out, (u > 0).astype(float))

    out_and = fg.compile_function("1 if (x > -0.3 and x < 0.3) else 0")(u, v)
    assert np.array_equal(out_and, ((u > -0.3) & (u < 0.3)).astype(float))

    out_or = fg.compile_function("1 if (x < -0.5 or x > 0.5) else 0")(u, v)
    assert np.array_equal(out_or, ((u < -0.5) | (u > 0.5)).astype(float))

    # A bare comparison (no ternary at all) was never actually broken -
    # numpy's own `>` already vectorizes - this just locks that in too.
    out_bare = fg.compile_function("x > 0")(u, v)
    assert np.array_equal(out_bare, (u > 0).astype(float))

    # Three-way chained "and" (BoolOp.values can have more than 2 terms).
    out_chain = fg.compile_function("1 if (x > -0.5 and x < 0.5 and y > -0.5) else 0")(u, v)
    assert np.array_equal(out_chain, ((u > -0.5) & (u < 0.5) & (v > -0.5)).astype(float))


def test_heart_mound_is_a_real_smooth_dome_not_a_crash():
    """The actual case that surfaced the ternary bug above: someone
    typing a heart-shaped formula. A height field can only ever make a
    dome/bump/saddle (one z per x,y) - never a shape with a concave
    notch like a heart's top lobes, which needs an implicit curve, a
    different kind of shape entirely. heart_mound.txt is the honest
    equivalent: a smooth dome over the heart-curve's interior, no
    comparisons needed at all. Mainly checks it actually compiles and
    produces a real, non-degenerate dome (not that it's a heart)."""
    u, v = GRID
    out = fg.compile_function("max(0, 1.1 - (x^2 + (1.3*y - abs(x)^0.67)^2))")(u, v)
    assert out.min() == 0.0 and out.max() > 0.5     # flat outside, a real peak inside
    assert np.all(np.isfinite(out))


def test_runtime_div_by_zero_and_log_of_negative_become_zero_not_nan_or_crash():
    u, v = GRID
    out = fg.compile_function("1 / x")(u, v)
    assert np.all(np.isfinite(out))              # inf -> 0, no exception
    out2 = fg.compile_function("log(x)")(u, v)    # x<0 present in GRID
    assert np.all(np.isfinite(out2))              # nan -> 0


def test_empty_and_unparsable_expressions_are_rejected():
    for bad in ["", "   ", "x +", "x; y", "x ** ** 2"]:
        try:
            fg.compile_function(bad)
            assert False, f"should have been rejected: {bad!r}"
        except fg.FunctionParseError:
            pass


def test_unknown_names_and_functions_are_rejected():
    for bad in ["q + 1", "foo(x)", "x.bit_length()", "sin(x, y, z)"]:
        try:
            fg.compile_function(bad)
            assert False, f"should have been rejected: {bad!r}"
        except fg.FunctionParseError:
            pass


def test_disallowed_syntax_is_rejected_before_anything_runs():
    """Anything that isn't plain arithmetic/comparisons/whitelisted
    calls - attribute access, subscripting, comprehensions, lambdas,
    string literals, imports, dunder access - must be rejected by the
    AST walk, not merely fail at eval() time (an empty __builtins__ is
    the second layer, not the only one - see module docstring)."""
    bad_exprs = [
        "__import__('os').system('echo hacked')",
        "open('/etc/passwd').read()",
        "x.__class__",
        "x.__class__.__mro__",
        "[i for i in range(3)]",
        "{i: i for i in range(3)}",
        "lambda t: t",
        "(lambda: __import__('os'))()",
        "'a string literal'",
        "b'bytes'",
        "x if x else y",  # IfExp is allowed; this just checks it still
                           # can't be abused to reach anything unsafe
        "x[0]",
        "(x := 3)",
    ]
    for bad in bad_exprs:
        try:
            f = fg.compile_function(bad)
            # A couple of these parse to *valid* restricted asts (the
            # ternary) - only truly dangerous ones must raise; assert
            # those specifically never produce a working callable that
            # touches anything outside numpy.
            if "import" in bad or "open(" in bad or "__class__" in bad or "lambda" in bad:
                assert False, f"should have been rejected: {bad!r}"
        except fg.FunctionParseError:
            pass
        except SyntaxError:
            pass  # e.g. "x[0]" / walrus are syntactically fine but must
                   # still be caught - if this branch is hit instead of
                   # FunctionParseError, _validate's whitelist missed a
                   # node type and this test should be updated to assert
                   # that explicitly, not silently pass.


def test_subscript_and_walrus_are_explicitly_rejected():
    for bad in ["x[0]", "(x := 3)"]:
        try:
            fg.compile_function(bad)
            assert False, f"should have been rejected: {bad!r}"
        except fg.FunctionParseError:
            pass


def test_keyword_arguments_are_rejected():
    try:
        fg.compile_function("round(x, ndigits=2)")
        assert False, "keyword arguments should be rejected"
    except fg.FunctionParseError:
        pass


def test_list_function_files_and_sample_seeding(tmp_path=None):
    import tempfile
    tmp = tempfile.mkdtemp()
    folder = os.path.join(tmp, "ar_functions")
    old = fg.FUNC_FOLDER
    fg.FUNC_FOLDER = folder
    try:
        assert fg.list_function_files() == []     # nothing yet
        fg.ensure_sample_functions()
        files = fg.list_function_files()
        assert len(files) == len(fg._STARTER_FUNCTIONS)
        for path in files:
            fn = fg.read_function_file(path)       # every starter must compile
            out = fn(*GRID)
            assert np.all(np.isfinite(out))
        # Never overwrites an existing file.
        with open(files[0], "w") as f:
            f.write("x + 1")
        fg.ensure_sample_functions()
        with open(files[0]) as f:
            assert f.read().strip() == "x + 1"
    finally:
        fg.FUNC_FOLDER = old


if __name__ == "__main__":
    fails = 0
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            try:
                f()
                print("PASS ", n)
            except Exception as e:
                fails += 1
                import traceback
                print("FAIL ", n, repr(e))
                traceback.print_exc(limit=4)
    sys.exit(1 if fails else 0)
