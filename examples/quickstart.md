# Quickstart

## 0. Preflight

```bash
./setup_and_run.sh                 # renderer only — prove the drawing half
./setup_and_run.sh --predict       # + tribev2 + torch
make doctor
```

Resolve anything marked ✗ before going further. The one that catches most
people is `hf:meta-llama/Llama-3.2-3B` — it's a gated repo, so you need to
accept the licence on the model page and then `hf auth login`.

## 1. Pull the weights up front

```bash
videocortex fetch
```

~0.7 GB for TRIBE v2 itself, plus roughly 15–20 GB across the four frozen
encoders. Doing this separately means a failed download doesn't waste an
inference run.

## 2. Start small

```bash
videocortex render --video some_clip.mp4 --max-frames 12
```

Use something **under 30 seconds** for your first run. On Metal the frozen
encoders dominate — V-JEPA2 ViT-g and a 3B language model over every chunk —
so a two-minute clip is not a two-minute wait.

Watch the log line that reports the device:

```
INFO videocortex.model: device: mps (Apple arm64 / Metal)
```

If that says `cpu` on an Apple Silicon machine, `videocortex doctor` will tell
you why (usually an x86_64 python under Rosetta).

## 3. Iterate on the picture, not the model

Inference already saved `predictions.npy`. Re-render for free:

```bash
videocortex overlay --run runs/some_clip
videocortex overlay --run runs/some_clip --spin
videocortex draw runs/some_clip/predictions.npy --views full
videocortex draw runs/some_clip/predictions.npy --light --threshold-frac 0.4
```

## 4. Read it honestly

- Predictions are for an **average subject**, not anyone in particular.
- They sit on the fsaverage5 cortical surface: 20,484 vertices, no subcortex.
- Upstream offsets predictions 5 s into the past for haemodynamic lag, so a
  frame corresponds to what happened roughly five seconds earlier.
- One TR ≈ 1.49 s. There is no finer temporal structure to read into it.

## 5. Command deck

```bash
videocortex serve                  # http://127.0.0.1:8730
./setup_and_run.sh --deck          # same, after the usual bootstrap
```

Local only. Doctor / Launch / Runs / Job. Launching an encode from the
browser still runs TRIBE on this machine — it is not a remote API.

## Try it without the model

The repo ships a synthetic run so you can exercise the renderer immediately:

```bash
make sample
open examples/sample_run/contact_sheet.png
```

`examples/make_sample.py` builds a seeded, scripted occipital → temporal
sequence — early visual cortex throughout, hMT+ waxing and waning, superior
temporal joining once someone starts talking. It is **not** model output. It
exists to prove the drawing half works on your machine before you commit to a
20 GB download.
