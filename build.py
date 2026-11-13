#!/usr/bin/env python3
"""Regenerate the generated build artifacts.

Two generated inputs, produced by two different toolchains:
  * `web/pkg/`                  — the WASM bundle (wasm-pack build of
    riffnet; both models' weights are embedded in it at compile
    time, read from `model/weights/riffnet.onnx` AND
    `model/weights/bass_riffnet.onnx` by riffnet/build.rs)
  * `model/weights/riffnet.onnx` / `bass_riffnet.onnx` — the ONNX exports
    (model/scripts/export.py, `--instrument bass` for the latter), the
    source build.rs reads from

Both are committed so the demo runs without any toolchain, but change the Rust
or retrain a model and you must regenerate them. This is the single entry
point that does that, so the outputs never drift.

Usage:
  python build.py                 # rebuild the WASM into web/pkg
  python build.py --ckpt PATH     # also re-export model/weights/riffnet.onnx (guitar), then rebuild wasm
  # to re-export bass: model/scripts/export.py --instrument bass, then `python build.py`
"""
import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def run(cmd, cwd=None, env=None):
    # Resolve the executable so Windows finds `wasm-pack.exe`/`.cmd` shims.
    exe = shutil.which(cmd[0])
    if exe is None:
        sys.exit(f"error: `{cmd[0]}` not found on PATH")
    print(f"$ {' '.join(cmd)}")
    subprocess.run([exe, *cmd[1:]], cwd=cwd or ROOT, env=env, check=True)


def path_remaps():
    """--remap-path-prefix flags for every local root whose absolute path
    rustc would otherwise compile into panic-location strings in the shipped
    .wasm (the repo checkout incl. build.rs's OUT_DIR, the cargo registry,
    and the rustup toolchain's rust-src). Without these the committed binary
    carries the builder's home directory and drive layout."""
    home = os.path.expanduser("~")
    roots = {
        ROOT: "/riffnet",
        os.environ.get("CARGO_HOME") or os.path.join(home, ".cargo"): "/cargo",
        os.environ.get("RUSTUP_HOME") or os.path.join(home, ".rustup"): "/rustup",
    }
    return [f"--remap-path-prefix={os.path.abspath(src)}={dst}"
            for src, dst in roots.items()]


def build_wasm():
    # CARGO_ENCODED_RUSTFLAGS (0x1f-separated) rather than RUSTFLAGS, so a
    # remapped path containing spaces stays one argument.
    flags = ["-C", "target-feature=+simd128", *path_remaps()]
    env = dict(os.environ, CARGO_ENCODED_RUSTFLAGS="\x1f".join(flags))
    env.pop("RUSTFLAGS", None)
    run(["wasm-pack", "build", "riffnet",
         "--target", "web", "--out-dir", "../web/pkg", "--release"], env=env)
    # wasm-pack drops a `.gitignore` (`*`) in the out-dir; remove it so the
    # generated pkg stays committed (the demo runs with no Rust toolchain).
    gi = os.path.join(ROOT, "web", "pkg", ".gitignore")
    if os.path.exists(gi):
        os.remove(gi)
    print("-> web/pkg regenerated")


def export_onnx(ckpt):
    run([sys.executable, "-m", "scripts.export", "--ckpt", ckpt, "--onnx", "weights/riffnet.onnx"],
        cwd=os.path.join(ROOT, "model"))
    print("-> model/weights/riffnet.onnx regenerated")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="", help="checkpoint to re-export model/weights/riffnet.onnx from")
    ap.add_argument("--skip-wasm", action="store_true", help="only re-export the onnx")
    args = ap.parse_args()

    # Export before building wasm: build.rs reads model/weights/riffnet.onnx
    # and bakes the weights into the wasm binary at compile time, so a stale
    # onnx would silently ship the old weights.
    if args.ckpt:
        export_onnx(args.ckpt)
    if not args.skip_wasm:
        build_wasm()
    print("done. reload the web app to pick up the new artifacts.")


if __name__ == "__main__":
    main()
