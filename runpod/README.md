# RunPod — Phase 3 training

End-to-end recipe to run AV+AR SFT warm-start on RunPod (recommended A5000 24GB at ~$0.27/hr; A100 SXM ~$1.49/hr if you want speed).

## One-time pod setup

1. **Create pod**:
   - Template: any PyTorch CUDA template (e.g. `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`).
   - GPU: RTX A5000 (cheapest viable) or A100 SXM (fastest reasonable).
   - Volume: 50 GB persistent. Mount path `/workspace`.
   - Expose SSH (port 22).

2. **Copy the data bundle** to the pod's persistent volume from your local machine:
   ```bash
   scp guppylm-nla-bundle.tar.gz root@<pod-host>:/workspace/
   ```
   The bundle (~114 MB) holds Phase 1 activations + Phase 2 summaries + GuppyLM checkpoint. The repo code is cloned fresh on the pod.

3. **SSH in** and run the setup script (one time):
   ```bash
   ssh root@<pod-host>
   cd /workspace
   git clone https://github.com/suvojit-0x55aa/guppylm-nla.git
   cd guppylm-nla
   bash runpod/setup.sh
   ```
   Setup clones the repo, installs Python deps, untars the bundle into the right places, and prints a one-line GPU check.

## Run smoke test (~3 min)

Verifies the pipeline end-to-end on real GPU before committing to the long run.

```bash
cd /workspace/guppylm-nla
bash runpod/smoke.sh
```

Expects: AV trains 30+ steps, AR trains 30+ steps, FVE eval finishes, final FVE prints. Smoke results in `/workspace/out/smoke/`.

## Run full training (convergence-bounded)

Once smoke passes, run both variants. Stops on convergence; effectively no time cap.

```bash
cd /workspace/guppylm-nla
bash runpod/full.sh
```

Trains AV+AR on `summary_text`, then on `summary_lens`, with checkpoints written to `/workspace/ckpts/phase3/{av,ar}_{text,lens}/final.pt` after each phase. Re-running `runpod/full.sh` will skip already-trained variants (looks for `final.pt`).

Results in `/workspace/out/`:
- `history_text.json`, `history_lens.json` — per-variant FVE curves and final numbers
- `MANIFEST_phase3.json` — provenance + decision banner
- `fve_curves.png` (if you run the plot cell — see `runpod/plot.sh`)

## Pull results back to your laptop

```bash
scp -r root@<pod-host>:/workspace/out ./phase3-out
scp -r root@<pod-host>:/workspace/ckpts ./phase3-ckpts   # optional, ~200 MB
```

Then **terminate the pod** to stop billing. The persistent volume is also chargeable; delete it if you don't need to resume.

## Cost guide

| GPU | $/hr | Smoke | Full text+lens |
|---|---|---|---|
| A5000 (Ampere) | $0.27 | ~$0.02 | ~$0.40–$0.80 |
| L4 (Ada)       | $0.39 | ~$0.02 | ~$0.50–$1.00 |
| RTX 4090       | $0.69 | ~$0.03 | ~$0.50–$0.80 |
| A100 SXM       | $1.49 | ~$0.04 | ~$0.50–$1.00 |
