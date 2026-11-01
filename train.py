from __future__ import annotations

import copy
import csv
import json
import math
import time
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
import torch

try:
    from .datasets import AUDIO_DATASETS, DATASETS, load_dataset
    from .losses import DIVERGENCES, align_loss, mse_loss
    from .model import SQMM
except ImportError:
    from datasets import AUDIO_DATASETS, DATASETS, load_dataset
    from losses import DIVERGENCES, align_loss, mse_loss
    from model import SQMM

# Main-table protocol per dataset: epochs and the selected growth interval G*.
DATASET_DEFAULTS = {
    "cello": {"epochs": 300, "growth": 25},
    "camera": {"epochs": 300, "growth": 25},
    "coffee": {"epochs": 300, "growth": 25},
    "astronaut": {"epochs": 600, "growth": 50},
}

DEFAULT_MODEL_CONFIG = {
    # PE frequencies; None picks the modality default (audio: dyadic 16, images: consecutive 8).
    "frequencies": None,
    "band_depths": [2, 3, 4, 4, 5],
    "band_widths": [2, 2, 2, 2, 2],
    "memory": {
        "mode": "read_write",      # read_write | read | write | none
        "qubits": 2,
        "init_axes": None,         # initialisation axis per memory qubit; None = RY, RX, ...
        "read_gates": None,        # per band; None = [cry] ... [cry, crx]
        "write_gates": None,       # per band; None = [xy] ... [xy], [] (the last band does not write)
    },
    "upload_axes": None,           # rotation axis per coordinate; None = RY (1-D), RY, RX (2-D)
    "readout": ["Z", "ZZ"],
    "readout_mlp": {"widths": [5], "activation": "gelu"},
}
MODALITY_FREQUENCIES = {
    "audio": {"mode": "dyadic", "amount": 16},
    "image": {"mode": "consecutive", "amount": 8},
}

DEFAULT_TRAIN_CONFIG = {
    "dataset": "cello",
    "epochs": None,                # None = DATASET_DEFAULTS
    "growth": None,                # updates between band introductions; None = G*; 0 = no growth
    "fade_updates": 5,
    "learning_rate": 0.03,
    "img_size": 32,
    "align_weight": 3.0,
    "align_divergence": "squared", # squared | kl
    "align_end_update": None,      # None = (num_bands - 1) * growth, the last band's introduction
}


def deep_update(base: dict, updates: dict | None) -> dict:
    result = copy.deepcopy(base)
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def resolve_train_config(train_config: dict | None = None) -> dict:
    config = deep_update(DEFAULT_TRAIN_CONFIG, train_config)
    unknown = set(config) - set(DEFAULT_TRAIN_CONFIG)
    if unknown:
        raise ValueError(f"unknown train_config keys: {sorted(unknown)}")
    if config["dataset"] not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}")
    for key, value in DATASET_DEFAULTS[config["dataset"]].items():
        if config[key] is None:
            config[key] = value
    if config["align_divergence"] not in DIVERGENCES:
        raise ValueError(f"align_divergence must be one of {DIVERGENCES}")
    return config


def build_model(model_config: dict | None, *, coord_dim: int, out_features: int) -> SQMM:
    config = deep_update(DEFAULT_MODEL_CONFIG, model_config)
    modality = "audio" if coord_dim == 1 else "image"
    return SQMM(
        coord_dim=coord_dim,
        out_features=out_features,
        frequencies=config["frequencies"] or MODALITY_FREQUENCIES[modality],
        band_depths=config["band_depths"],
        band_widths=config["band_widths"],
        memory=config["memory"],
        upload_axes=config["upload_axes"],
        readout=config["readout"],
        readout_mlp=config["readout_mlp"],
    )


def growth_state(update: int, growth: int, num_bands: int, fade_updates: int = 5):
    """(active bands, fade of the newest band) at an optimizer update."""
    if growth <= 0:
        return num_bands, 1.0
    active = min(1 + update // growth, num_bands)
    if active == 1 or fade_updates == 0:
        return active, 1.0
    return active, min(1.0, float(update - (active - 1) * growth + 1) / float(fade_updates))


def batch_loss(model, coords, target, grid_shape, config, *, update: int, active: int, fade: float):
    """Training loss of one full-batch update. Returns (loss, task MSE)."""
    end = config["align_end_update"]
    if end is None and config["growth"] > 0:
        end = (model.num_bands - 1) * config["growth"]
    use_align = active >= 2 and (end is None or update < end)

    if use_align:
        prediction, band_features, memory_states = model(coords, active, fade, return_memory=True)
    else:
        prediction = model(coords, active, fade)
    mse = mse_loss(prediction, target)
    loss = mse
    if use_align:
        aligned = align_loss(band_features, memory_states, grid_shape, config["align_divergence"])
        loss = loss + config["align_weight"] * aligned
    return loss, mse


@torch.no_grad()
def evaluate(model, coords, target, *, image: bool) -> dict:
    """Full-grid scores. MSE is in the training range [-1, 1]; image PSNR/SSIM use [0, 1]."""
    prediction = model(coords)
    mse = float(((prediction - target) ** 2).mean())
    if not image:
        return {"mse": mse, "psnr": 10.0 * math.log10(1.0 / max(mse, 1e-12))}
    unit_prediction = (prediction.clamp(-1.0, 1.0) + 1.0) * 0.5
    unit_target = (target.clamp(-1.0, 1.0) + 1.0) * 0.5
    unit_mse = float(((unit_prediction - unit_target) ** 2).mean())
    metrics = {"mse": mse, "psnr": 10.0 * math.log10(1.0 / max(unit_mse, 1e-12))}
    try:
        from skimage.metrics import structural_similarity

        side = int(round(math.sqrt(target.shape[1])))
        images = [t.reshape(side, side).numpy().astype(np.float32) for t in (unit_target[0, :, 0], unit_prediction[0, :, 0])]
        metrics["ssim"] = float(structural_similarity(*images, data_range=1.0))
    except ImportError:
        metrics["ssim"] = None
    return metrics


def train_one_run(*, run_id: int = 0, model_config: dict | None = None, train_config: dict | None = None,
                  verbose: bool = True) -> dict:
    config = resolve_train_config(train_config)
    dataset = config["dataset"]
    coords, target, grid_shape = load_dataset(dataset, config["img_size"])
    model = build_model(model_config, coord_dim=coords.shape[-1], out_features=target.shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], betas=(0.9, 0.999), eps=1e-8)

    best_mse, best_epoch, best_state = float("inf"), None, None
    started = time.time()
    for epoch in range(config["epochs"]):
        update = epoch  # one full-batch update per epoch
        active, fade = growth_state(update, config["growth"], model.num_bands, config["fade_updates"])
        # The epoch's MSE is measured before its update, so it scores these weights.
        state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        loss, mse = batch_loss(model, coords, target, grid_shape, config, update=update, active=active, fade=fade)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_mse = mse.item()
        if active == model.num_bands and fade >= 1.0 and epoch_mse < best_mse:
            best_mse, best_epoch, best_state = epoch_mse, epoch, state
        if verbose and epoch % 50 == 0:
            print(f"{dataset} run {run_id} | epoch {epoch:4d}/{config['epochs']} | "
                  f"bands {active} | mse {epoch_mse:.6f} | best {best_mse:.6f}", flush=True)

    image = dataset not in AUDIO_DATASETS
    final = evaluate(model, coords, target, image=image)
    final_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    best = evaluate(model, coords, target, image=image)
    result = {
        "dataset": dataset, "run_id": int(run_id),
        "epochs": config["epochs"], "growth": config["growth"], "best_epoch": best_epoch,
        **{f"{key}_best": value for key, value in best.items()},
        **{f"{key}_final": value for key, value in final.items()},
        **model.parameter_counts(),
        "elapsed_s": time.time() - started,
    }
    if verbose:
        print(f"{dataset} run {run_id} | DONE | mse_best={result['mse_best']:.6e} "
              f"psnr_best={result['psnr_best']:.2f} | params={result['trainable_params']}", flush=True)
    return {"result": result, "best_state": model.state_dict(), "final_state": final_state}


### Multi-run driver and outputs ###


def summarize(rows: list[dict]) -> dict:
    summary = {}
    for dataset in sorted({r["dataset"] for r in rows}):
        group = [r for r in rows if r["dataset"] == dataset]
        mses = [r["mse_best"] for r in group]
        psnrs = [r["psnr_best"] for r in group]
        summary[dataset] = {
            "runs": len(group),
            "mean_mse_best": mean(mses), "std_mse_best": pstdev(mses), "min_mse_best": min(mses),
            "mean_psnr_best": mean(psnrs), "max_psnr_best": max(psnrs),
        }
    return summary


def write_outputs(output_dir: Path, rows: list[dict], configs: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output_dir / "runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    payload = {**configs, "summary": summarize(rows), "runs": rows}
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2))


def run(*, runs: int, output_dir: str | Path, model_config: dict | None = None,
        train_config: dict | None = None, save_weights: bool = False, verbose: bool = True) -> dict:
    config = resolve_train_config(train_config)
    output_dir = Path(output_dir)
    rows = []
    for run_id in range(runs):
        out = train_one_run(run_id=run_id, model_config=model_config, train_config=config, verbose=verbose)
        rows.append(out["result"])
        if save_weights:
            weights = output_dir / "weights"
            weights.mkdir(parents=True, exist_ok=True)
            tag = f"{config['dataset']}_run{run_id}"
            torch.save(out["best_state"], weights / f"{tag}_best.pt")
            torch.save(out["final_state"], weights / f"{tag}_final.pt")
        write_outputs(output_dir, rows, {"model_config": deep_update(DEFAULT_MODEL_CONFIG, model_config),
                                         "train_config": config})
    return {"summary": summarize(rows), "runs": rows}
