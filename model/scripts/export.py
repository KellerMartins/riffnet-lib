"""Export RiffNet to ONNX with a fixed streaming window (T=WIN frames), and dump
a test input + reference outputs so the Rust side can verify parity.

At inference the Rust side feeds the last WIN CQT frames (1,N_IN,WIN,NUM_BINS)
and reads the LAST time column of each output head.
"""
import argparse
import numpy as np
import torch
from model import (RiffNet, RiffNetBass, load_model, load_model_bass,
                    NUM_BINS, BASS_NUM_BINS, N_IN)

WIN = 17  # streaming window length (>= receptive field of 17)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instrument", default="guitar", choices=["guitar", "bass"])
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--onnx", default="")
    ap.add_argument("--dump", default="parity")
    args = ap.parse_args()
    is_bass = args.instrument == "bass"
    ckpt_path = args.ckpt or ("weights/bass_riffnet.pt" if is_bass else "weights/riffnet.pt")
    onnx_path = args.onnx or ("weights/bass_riffnet.onnx" if is_bass else "weights/riffnet.onnx")
    num_bins = BASS_NUM_BINS if is_bass else NUM_BINS

    torch.manual_seed(0)
    if is_bass:
        model = load_model_bass(torch.load(ckpt_path, map_location="cpu")) if ckpt_path else RiffNetBass()
    else:
        model = load_model(torch.load(ckpt_path, map_location="cpu")) if ckpt_path else RiffNet()
    model.eval()

    dummy = torch.zeros(1, N_IN, WIN, num_bins)   # HCQT: N_IN harmonic channels
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["cqt"], output_names=["onset", "pitch"],
        opset_version=17, dynamo=False,
        # flexible time axis: at inference we feed exactly WIN=17 frames -> 1 output
        dynamic_axes={"cqt": {2: "T"}, "onset": {1: "Tout"}, "pitch": {1: "Tout"}},
    )
    
    # Run ORT Graph Optimization to fuse Conv + BatchNorm + ReLU
    import onnxruntime as ort
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess_options.optimized_model_filepath = onnx_path
    _ = ort.InferenceSession(onnx_path, sess_options)

    print(f"exported optimized {onnx_path} (input 1x{N_IN}x{WIN}x{num_bins}, dynamic T)")

    # Reference input/outputs for parity (use a deterministic non-trivial input)
    rng = np.random.default_rng(42)
    x = (rng.random((1, N_IN, WIN, num_bins), dtype=np.float32) * 0.03)
    with torch.no_grad():
        on_l, pi_l = model(torch.from_numpy(x))
    onset = torch.sigmoid(on_l)[0, -1].item()           # last frame
    pitch = torch.sigmoid(pi_l)[0, -1].numpy()          # last frame, num_pitch classes
    dump_prefix = args.dump if args.dump != "parity" else ("bass_parity" if is_bass else "parity")
    x.tofile(f"{dump_prefix}_input.bin")
    np.array([onset], dtype=np.float32).tofile(f"{dump_prefix}_onset.bin")
    pitch.astype(np.float32).tofile(f"{dump_prefix}_pitch.bin")
    print(f"parity dump: last-frame onset={onset:.5f}, pitch[:5]={pitch[:5]}")


if __name__ == "__main__":
    main()
