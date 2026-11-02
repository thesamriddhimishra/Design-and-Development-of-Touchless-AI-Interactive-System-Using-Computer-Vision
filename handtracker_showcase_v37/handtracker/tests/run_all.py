"""Run every test suite: python tests/run_all.py   (no camera/display needed)"""
import os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SUITES = ["test_ar3d.py", "test_ar_overlay_integration.py", "test_function_graphs.py",
          "test_game.py", "test_main_overlay.py", "test_overlay_drawing.py", "test_tutorial.py",
          "test_video_call.py", "test_render_equivalence.py", "test_perf.py", "test_smoke.py"]
bad = 0
for s in SUITES:
    p = subprocess.run([sys.executable, os.path.join(HERE, s)], capture_output=True, text=True)
    passed = p.stdout.count("PASS")
    status = "ok " if p.returncode == 0 else "FAIL"
    print(f"{status} {s:36s} {passed} passed")
    if p.returncode:
        bad += 1
        print((p.stdout + p.stderr)[-1500:])
sys.exit(1 if bad else 0)
