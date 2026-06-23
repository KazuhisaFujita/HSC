# train_hsc.py
from pathlib import Path
import csv
from dataclasses import dataclass
import json
from statistics import median
import time

import torch
import torch.nn as nn

from HierarchicalSparseCoding import (
    Config,
    normalize_dictionary_,
    sparsity_stats,
    make_run_dir,
    expand_param_list,
    UnifiedHSCInference,
    compute_hsc_layer_steps,
    hierarchical_losses,
)

from datasets import load_dataset, normalize_dataset_name as normalize_dataset_key


SUPPORTED_DATASETS = (
    "mnist",
    "kmnist",
    "fashion_mnist",
    "cifar10",
    "cifar10_gray",
    "bsds500",
    "bsds500_patch",
    "parity",
)


@dataclass
class TrainedRunArtifacts:
    run_dir: Path
    run_name: str
    dataset_name: str
    Ds: nn.ParameterList
    infer_module: UnifiedHSCInference

def init_csv(path: Path, fieldnames: list):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def append_csv(path: Path, row: dict):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)


def save_latency_eval_csv(run_dir: Path, row: dict):
    latency_path = run_dir / "latency_eval.csv"
    init_csv(latency_path, list(row.keys()))
    append_csv(latency_path, row)


def save_latency_eval_batch_csv(run_dir: Path, rows: list[dict]):
    if not rows:
        return
    latency_path = run_dir / "latency_eval_batches.csv"
    init_csv(latency_path, list(rows[0].keys()))
    for row in rows:
        append_csv(latency_path, row)


def save_test_metrics_csv(run_dir: Path, row: dict):
    metrics_path = run_dir / "test_metrics.csv"
    init_csv(metrics_path, list(row.keys()))
    append_csv(metrics_path, row)


def save_config_csv(run_dir: Path, cfg: Config, dataset_name: str, seed: int | None = None):
    config_path = run_dir / "config.csv"

    row = {
        "dataset_name": dataset_name,
        "seed": seed,
        "mode": cfg.mode,
        "lista_variant": cfg.lista_variant,
        "layer_dims": json.dumps(cfg.layer_dims),
        "lambdas": json.dumps(
            list(cfg.lambdas) if isinstance(cfg.lambdas, (list, tuple)) else [cfg.lambdas]
        ),
        "betas": json.dumps(
            list(cfg.betas) if isinstance(cfg.betas, (list, tuple)) else [cfg.betas]
        ),
        "lr_D": cfg.lr_D,
        "lr_E": cfg.lr_E,
        "infer_steps": cfg.infer_steps,
        "lista_steps": cfg.lista_steps,
        "eta_scale": cfg.eta_scale,
        "hybrid_pretrain_epochs": cfg.hybrid_pretrain_epochs,
        "batch_size": cfg.batch_size,
        "epochs": cfg.epochs,
        "dc_center": cfg.dc_center,
        "device": cfg.device,
    }

    init_csv(config_path, list(row.keys()))
    append_csv(config_path, row)


def normalize_dataset_name(dataset_name: str) -> str:
    return normalize_dataset_key(dataset_name)


def make_dataset_loaders(dataset_name: str, batch_size: int):
    normalized_name = normalize_dataset_name(dataset_name)

    try:
        train_loader, val_loader, test_loader = load_dataset(
            dataset_type=normalized_name,
            batch_size=batch_size,
            flatten=False,
        )
    except ValueError as exc:
        supported = ", ".join(SUPPORTED_DATASETS)
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Choose one of: {supported}."
        ) from exc

    return normalized_name, train_loader, val_loader, test_loader


def infer_input_layout(images: torch.Tensor):
    if images.ndim == 4:
        _, channels, height, width = images.shape
        input_dim = channels * height * width
        return input_dim, height, width, channels == 1

    if images.ndim == 3:
        _, height, width = images.shape
        input_dim = height * width
        return input_dim, height, width, True

    if images.ndim == 2:
        return images.shape[1], None, None, False

    raise ValueError(f"Unsupported input batch shape: {tuple(images.shape)}")


def make_train_log_fields(n_layers: int):
    fields = [
        "run_name",
        "mode",
        "lista_variant",
        "epoch",
        "global_step",
        "wall_time_sec",
        "infer_time_ms",
        "loss",
        "rec_x",
        "rec_h",
        "sparse",
    ]
    for level in range(1, n_layers + 1):
        fields += [
            f"a{level}_l1_mean",
            f"a{level}_active_frac",
        ]
    return fields


def make_epoch_metrics_fields(n_layers: int):
    fields = [
        "run_name",
        "mode",
        "lista_variant",
        "epoch",
        "n_layers",
        "layer_dims",
        "loss",
        "rec_x",
        "rec_h",
        "sparse",
        "infer_time_ms",
        "num_params_dict",
        "num_params_encoder",
        "save_dir",
    ]
    for level in range(1, n_layers + 1):
        fields += [
            f"a{level}_l1_mean",
            f"a{level}_active_frac",
        ]
    return fields


def make_train_log_row(
    run_name: str,
    cfg: Config,
    epoch: int,
    global_step: int,
    wall_time_sec: float,
    infer_time_ms: float,
    loss: float,
    rec_x: float,
    rec_h: float,
    sparse: float,
    codes: list,
):
    row = {
        "run_name": run_name,
        "mode": cfg.mode,
        "lista_variant": cfg.lista_variant,
        "epoch": epoch,
        "global_step": global_step,
        "wall_time_sec": wall_time_sec,
        "infer_time_ms": infer_time_ms,
        "loss": loss,
        "rec_x": rec_x,
        "rec_h": rec_h,
        "sparse": sparse,
    }

    for level, a in enumerate(codes, start=1):
        st = sparsity_stats(a)
        row[f"a{level}_l1_mean"] = st["l1_mean"]
        row[f"a{level}_active_frac"] = st["active_frac"]

    return row


def make_epoch_metrics_row(
    run_name: str,
    cfg: Config,
    epoch: int,
    codes: list,
    loss: float,
    rec_x: float,
    rec_h: float,
    sparse: float,
    infer_time_ms: float,
    Ds,
    infer_module,
    run_dir: Path,
    code_stats: list[dict] | None = None,
):
    enc_params = infer_module.encoder_parameters() # LISTA encoder のパラメータ取得
    num_params_encoder = sum(p.numel() for p in enc_params) if len(enc_params) > 0 else 0

    row = {
        "run_name": run_name,
        "mode": cfg.mode,
        "lista_variant": cfg.lista_variant,
        "epoch": epoch,
        "n_layers": len(cfg.layer_dims),
        "layer_dims": json.dumps(cfg.layer_dims),
        "loss": loss,
        "rec_x": rec_x,
        "rec_h": rec_h,
        "sparse": sparse,
        "infer_time_ms": infer_time_ms,
        "num_params_dict": sum(D.numel() for D in Ds),
        "num_params_encoder": num_params_encoder,
        "save_dir": str(run_dir),
    }

    for level, a in enumerate(codes, start=1):
        st = code_stats[level - 1] if code_stats is not None else sparsity_stats(a)
        row[f"a{level}_l1_mean"] = st["l1_mean"]
        row[f"a{level}_active_frac"] = st["active_frac"]

    return row


def sync_if_needed(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def resolve_eval_loader(cfg: Config, data_loader=None, dataset_name="MNIST"):
    if isinstance(data_loader, str):
        dataset_name = data_loader
        data_loader = None

    normalized_dataset_name = normalize_dataset_name(dataset_name)

    if data_loader is None:
        normalized_dataset_name, _, _, data_loader = make_dataset_loaders(dataset_name, cfg.batch_size)

    return normalized_dataset_name, data_loader


def prepare_eval_batch(cfg: Config, img: torch.Tensor):
    img = img.to(cfg.device)
    x = img.view(img.size(0), -1)

    if cfg.dc_center:
        x = x - x.mean(dim=1, keepdim=True)

    return x


def build_eval_inference_context(cfg: Config, Ds):
    Ds_for_inference = [D.detach() for D in Ds] # 推論中は辞書を固定する
    betas = expand_param_list(cfg.betas, max(len(cfg.layer_dims) - 1, 0), "betas")
    lambdas = expand_param_list(cfg.lambdas, len(cfg.layer_dims), "lambdas")
    etas_eval = compute_hsc_layer_steps(
        Ds=Ds_for_inference,
        betas=betas,
        eta_scale=cfg.eta_scale,
    )
    return Ds_for_inference, lambdas, betas, etas_eval


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be between 0.0 and 1.0")
    sorted_values = sorted(float(value) for value in values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = (len(sorted_values) - 1) * q
    lower_index = int(index)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = index - lower_index
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    return lower_value + (upper_value - lower_value) * fraction


def make_latency_eval_row(
    ms_per_batch: float,
    ms_per_sample: float,
    median_ms_per_batch: float,
    median_ms_per_sample: float,
    p25_ms_per_batch: float,
    p75_ms_per_batch: float,
    p25_ms_per_sample: float,
    p75_ms_per_sample: float,
    num_batches: int,
    num_samples: int,
    warmup_batches: int,
):
    return {
        "latency_eval_ms_per_batch": ms_per_batch,
        "latency_eval_ms_per_sample": ms_per_sample,
        "latency_eval_median_ms_per_batch": median_ms_per_batch,
        "latency_eval_median_ms_per_sample": median_ms_per_sample,
        "latency_eval_p25_ms_per_batch": p25_ms_per_batch,
        "latency_eval_p75_ms_per_batch": p75_ms_per_batch,
        "latency_eval_p25_ms_per_sample": p25_ms_per_sample,
        "latency_eval_p75_ms_per_sample": p75_ms_per_sample,
        "latency_eval_num_batches": num_batches,
        "latency_eval_num_samples": num_samples,
        "latency_eval_warmup_batches": warmup_batches,
    }


@torch.no_grad()
def collect_evaluation_snapshot(
    cfg: Config,
    Ds,
    infer_module,
    data_loader,
    run_name: str,
    epoch: int,
    run_dir: Path,
    split_name: str,
):
    Ds_for_inference, lambdas, betas, etas_eval = build_eval_inference_context(cfg, Ds)

    total_loss = 0.0
    total_rec_x = 0.0
    total_rec_h = 0.0
    total_sparse = 0.0
    total_infer_time_ms = 0.0
    total_samples = 0
    total_batches = 0
    codes = None
    level_l1_sums = [0.0 for _ in cfg.layer_dims]
    level_active_sums = [0.0 for _ in cfg.layer_dims]

    was_training = infer_module.training
    infer_module.eval()

    for img, _ in data_loader:
        x = prepare_eval_batch(cfg, img)

        sync_if_needed(cfg.device)
        t0_infer = time.time()

        codes = infer_module(
            x=x,
            Ds_for_inference=Ds_for_inference,
            lambdas=lambdas,
            betas=betas,
            infer_steps=cfg.infer_steps,
            eta_scale=cfg.eta_scale,
            etas=etas_eval,
        )

        sync_if_needed(cfg.device)
        infer_time_ms = 1000.0 * (time.time() - t0_infer)

        # 最終 loss は元の learnable dictionaries で評価する
        loss, rec_x, rec_h, sparse = hierarchical_losses(
            x=x,
            Ds=list(Ds),
            codes=codes,
            lambdas=lambdas,
            betas=betas,
        )

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_rec_x += rec_x.item() * bs
        total_rec_h += rec_h.item() * bs
        total_sparse += sparse.item() * bs
        total_infer_time_ms += infer_time_ms
        total_samples += bs
        total_batches += 1

        for level, a in enumerate(codes):
            st = sparsity_stats(a)
            level_l1_sums[level] += st["l1_mean"] * bs
            level_active_sums[level] += st["active_frac"] * bs

    if total_samples == 0 or total_batches == 0 or codes is None:
        raise ValueError(f"{split_name} loader did not yield any batches.")

    avg_loss = total_loss / total_samples
    avg_rec_x = total_rec_x / total_samples
    avg_rec_h = total_rec_h / total_samples
    avg_sparse = total_sparse / total_samples
    avg_infer_time_ms = total_infer_time_ms / total_batches

    aggregated_code_stats = []
    for level in range(len(cfg.layer_dims)):
        aggregated_code_stats.append(
            {
                "l1_mean": level_l1_sums[level] / total_samples,
                "active_frac": level_active_sums[level] / total_samples,
            }
        )

    row = make_epoch_metrics_row(
        run_name=run_name,
        cfg=cfg,
        epoch=epoch,
        codes=codes,
        loss=avg_loss,
        rec_x=avg_rec_x,
        rec_h=avg_rec_h,
        sparse=avg_sparse,
        infer_time_ms=avg_infer_time_ms,
        Ds=Ds,
        infer_module=infer_module,
        run_dir=run_dir,
        code_stats=aggregated_code_stats,
    )
    row["split"] = split_name
    if was_training:
        infer_module.train()
    return row


def collect_validation_snapshot(
    cfg: Config,
    Ds,
    infer_module,
    val_loader,
    run_name: str,
    epoch: int,
    run_dir: Path,
):
    row = collect_evaluation_snapshot(
        cfg=cfg,
        Ds=Ds,
        infer_module=infer_module,
        data_loader=val_loader,
        run_name=run_name,
        epoch=epoch,
        run_dir=run_dir,
        split_name="validation",
    )
    return {key: value for key, value in row.items() if key != "split"}


def train_main(cfg: Config, dataset_name="MNIST", seed: int | None = None):
    # 辞書学習
    # mode の意味:
    #   ista            : 辞書を固定して ISTA 反復だけで潜在コードを推論する。
    #   mfista          : ISTA の代わりに単調 FISTA で潜在コードを推論する。
    #   lista           : 学習された LISTA-style encoder だけで潜在コードを推論する。
    #   hybrid          : LISTA 初期化の後に ISTA refinement を行う主 Hybrid 手法。
    #   hybrid_finetune : 最初の hybrid_pretrain_epochs だけ LISTA-only で学習し、
    #                     その後は hybrid と同じ推論経路を使う訓練スケジュール。
    #                     論文の報告対象からは外している。
    #   hybrid_mfista   : LISTA 初期化の後に MFISTA refinement を行う補助的変種。
    assert cfg.mode in ["ista", "mfista", "lista", "hybrid", "hybrid_finetune", "hybrid_mfista"]
    assert cfg.lista_variant in ["shared", "untied"]
    if cfg.mode == "hybrid_finetune":
        if cfg.hybrid_pretrain_epochs <= 0:
            raise ValueError("hybrid_pretrain_epochs must be > 0 when mode='hybrid_finetune'.")
        if cfg.hybrid_pretrain_epochs > cfg.epochs:
            raise ValueError("hybrid_pretrain_epochs must be <= epochs when mode='hybrid_finetune'.")

    run_dir = make_run_dir(cfg.save_root, prefix=f"hsc_{cfg.mode}_{cfg.lista_variant}")
    print(f"save_dir = {run_dir}")

    run_name = run_dir.name

    normalized_dataset_name, loader, val_loader, test_loader = make_dataset_loaders(dataset_name, cfg.batch_size)
    save_config_csv(run_dir, cfg, normalized_dataset_name, seed=seed)

    sample_images, _ = next(iter(loader))
    input_dim, _, _, _ = infer_input_layout(sample_images)
    n_layers = len(cfg.layer_dims)

    if n_layers < 1:
        raise ValueError("layer_dims must contain at least one layer.")

    lambdas = expand_param_list(cfg.lambdas, n_layers, "lambdas")
    betas = expand_param_list(cfg.betas, max(n_layers - 1, 0), "betas")

    # -------------------------------------------------
    # Logs
    # -------------------------------------------------
    train_log_path = run_dir / "train_log.csv"
    init_csv(train_log_path, make_train_log_fields(n_layers))

    # Epoch-level monitoring over the full validation loader.
    epoch_metrics_path = run_dir / "epoch_metrics.csv"
    init_csv(epoch_metrics_path, make_epoch_metrics_fields(n_layers))

    start_time = time.time()

    # -------------------------------------------------
    # Dictionaries D1...DL
    # -------------------------------------------------
    dims = [input_dim] + list(cfg.layer_dims)
    Ds = nn.ParameterList()

    for ell in range(n_layers):
        D = nn.Parameter(torch.randn(dims[ell], dims[ell + 1], device=cfg.device) * 0.05)
        normalize_dictionary_(D.data)
        Ds.append(D)

    # -------------------------------------------------
    # Unified inference module
    # -------------------------------------------------
    infer_module = UnifiedHSCInference(
        mode=cfg.mode,
        input_dim=input_dim,
        layer_dims=cfg.layer_dims,
        lista_steps=cfg.lista_steps,
        lista_variant=cfg.lista_variant,
    ).to(cfg.device)

    infer_module.init_from_dictionaries(
        Ds=[D.detach() for D in Ds],
        lambdas=lambdas,
        eta_scale=cfg.eta_scale,
    )

    # -------------------------------------------------
    # Optimizers
    # -------------------------------------------------
    opt_D = torch.optim.Adam(Ds.parameters(), lr=cfg.lr_D)

    enc_params = infer_module.encoder_parameters() # LISTA（Hybridも含む）を使う場合、エンコーダパラメータが入る
    opt_E = torch.optim.Adam(enc_params, lr=cfg.lr_E) if len(enc_params) > 0 else None # エンコーダパラメータがある場合のみオプティマイザを作る

    global_step = 0

    if cfg.epochs == 0:
        epoch_row = collect_validation_snapshot(
            cfg=cfg,
            Ds=Ds,
            infer_module=infer_module,
            val_loader=val_loader,
            run_name=run_name,
            epoch=0,
            run_dir=run_dir,
        )
        append_csv(epoch_metrics_path, epoch_row)

    for ep in range(1, cfg.epochs + 1):
        pretrain_phase = cfg.mode == "hybrid_finetune" and ep <= cfg.hybrid_pretrain_epochs
        for img, _ in loader:
            img = img.to(cfg.device)
            x = img.view(img.size(0), -1)

            if cfg.dc_center:
                x = x - x.mean(dim=1, keepdim=True)

            # -------------------------------------------------
            # Inference
            # training 中の infer_time_ms には step-size 推定時間も含まれる
            # -------------------------------------------------
            Ds_for_inference = [D.detach() for D in Ds] # 推論中は辞書を固定する

            sync_if_needed(cfg.device)
            t0_infer = time.time()

            if pretrain_phase:
                codes = infer_module.lista(x)
            else:
                codes = infer_module(
                    x=x,
                    Ds_for_inference=Ds_for_inference,
                    lambdas=lambdas,
                    betas=betas,
                    infer_steps=cfg.infer_steps,
                    eta_scale=cfg.eta_scale,
                )

            sync_if_needed(cfg.device)
            infer_time_ms = 1000.0 * (time.time() - t0_infer)

            # -------------------------------------------------
            # Loss, 最終 loss は元の learnable dictionaries で評価する
            # -------------------------------------------------
            loss, rec_x, rec_h, sparse = hierarchical_losses(
                x=x,
                Ds=list(Ds),
                codes=codes,
                lambdas=lambdas,
                betas=betas,
            )

            # -------------------------------------------------
            # Update
            # 辞書更新は inference-through-D ではなく final-loss-through-D
            # -------------------------------------------------
            opt_D.zero_grad(set_to_none=True)
            if opt_E is not None:
                opt_E.zero_grad(set_to_none=True)

            loss.backward()

            opt_D.step() # optimizer step
            if opt_E is not None:
                opt_E.step() # optimizer step

            # 正規化制約
            # 更新後の各辞書列を L2 正規化します。normalize_dictionary_ は列ごとにノルムを計算し、D.div_(norms) で in-place に正規化しています。
            for D in Ds:
                normalize_dictionary_(D.data)

            # -------------------------------------------------
            # Monitoring
            # -------------------------------------------------
            if global_step % cfg.print_every == 0:
                row = make_train_log_row(
                    run_name=run_name,
                    cfg=cfg,
                    epoch=ep,
                    global_step=global_step,
                    wall_time_sec=time.time() - start_time,
                    infer_time_ms=infer_time_ms,
                    loss=loss.item(),
                    rec_x=rec_x.item(),
                    rec_h=rec_h.item(),
                    sparse=sparse.item(),
                    codes=codes,
                )
                append_csv(train_log_path, row)

                phase_label = "lista" if pretrain_phase else cfg.mode
                msg = (
                    f"mode {cfg.mode:>15s} | "
                    f"phase {phase_label:>15s} | "
                    f"ep {ep:02d} step {global_step:06d} | "
                    f"energy {loss.item():.4f} | "
                    f"rec_x {rec_x.item():.4f} | "
                    f"rec_h {rec_h.item():.4f} | "
                    f"sparse {sparse.item():.4f}"
                )
                for level, a in enumerate(codes, start=1):
                    st = sparsity_stats(a)
                    msg += f" | a{level}: |.|_1 {st['l1_mean']:.4f}, active {st['active_frac']:.4f}"
                print(msg)

            global_step += 1

        # -------------------------------------------------
        # Epoch-end monitoring over the full validation loader
        # -------------------------------------------------
        with torch.no_grad():
            epoch_row = collect_validation_snapshot(
                cfg=cfg,
                Ds=Ds,
                infer_module=infer_module,
                val_loader=val_loader,
                run_name=run_name,
                epoch=ep,
                run_dir=run_dir,
            )
            append_csv(epoch_metrics_path, epoch_row)

    with torch.no_grad():
        test_row = collect_evaluation_snapshot(
            cfg=cfg,
            Ds=Ds,
            infer_module=infer_module,
            data_loader=test_loader,
            run_name=run_name,
            epoch=cfg.epochs,
            run_dir=run_dir,
            split_name="test",
        )
        save_test_metrics_csv(run_dir, test_row)

    print("done")
    print(f"results saved in: {run_dir}")

    return TrainedRunArtifacts(
        run_dir=run_dir,
        run_name=run_name,
        dataset_name=normalized_dataset_name,
        Ds=Ds,
        infer_module=infer_module,
    )


@torch.no_grad()
def measure_inference_latency(
    cfg: Config,
    Ds,
    infer_module,
    data_loader=None,
    dataset_name="MNIST",
    warmup_batches: int = 1,
    max_batches: int | None = None,
):
    infer_module.eval()

    if warmup_batches < 0:
        raise ValueError("warmup_batches must be >= 0")
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be > 0 when provided")

    _, data_loader = resolve_eval_loader(cfg, data_loader=data_loader, dataset_name=dataset_name)
    Ds_for_inference, lambdas, betas, etas_eval = build_eval_inference_context(cfg, Ds)

    total_time_ms = 0.0
    total_samples = 0
    total_batches = 0
    batch_times_ms = []
    sample_times_ms = []
    batch_rows = []

    for batch_index, (img, _) in enumerate(data_loader):
        x = prepare_eval_batch(cfg, img)

        if batch_index < warmup_batches:
            infer_module(
                x=x,
                Ds_for_inference=Ds_for_inference,
                lambdas=lambdas,
                betas=betas,
                infer_steps=cfg.infer_steps,
                eta_scale=cfg.eta_scale,
                etas=etas_eval,
            )
            continue

        if max_batches is not None and total_batches >= max_batches:
            break

        sync_if_needed(cfg.device)
        t0 = time.time()

        infer_module(
            x=x,
            Ds_for_inference=Ds_for_inference,
            lambdas=lambdas,
            betas=betas,
            infer_steps=cfg.infer_steps,
            eta_scale=cfg.eta_scale,
            etas=etas_eval,
        )

        sync_if_needed(cfg.device)
        dt_ms = 1000.0 * (time.time() - t0)
        per_sample_ms = dt_ms / x.size(0)

        total_time_ms += dt_ms
        total_batches += 1
        total_samples += x.size(0)
        batch_times_ms.append(dt_ms)
        sample_times_ms.append(per_sample_ms)
        batch_rows.append(
            {
                "timed_batch_index": total_batches,
                "source_batch_index": batch_index,
                "batch_size": x.size(0),
                "latency_eval_ms_per_batch": dt_ms,
                "latency_eval_ms_per_sample": per_sample_ms,
            }
        )

    if total_batches == 0 or total_samples == 0:
        raise ValueError("Latency evaluation did not measure any batches. Reduce warmup_batches or increase data size.")

    return make_latency_eval_row(
        ms_per_batch=total_time_ms / total_batches,
        ms_per_sample=total_time_ms / total_samples,
        median_ms_per_batch=median(batch_times_ms),
        median_ms_per_sample=median(sample_times_ms),
        p25_ms_per_batch=percentile(batch_times_ms, 0.25),
        p75_ms_per_batch=percentile(batch_times_ms, 0.75),
        p25_ms_per_sample=percentile(sample_times_ms, 0.25),
        p75_ms_per_sample=percentile(sample_times_ms, 0.75),
        num_batches=total_batches,
        num_samples=total_samples,
        warmup_batches=warmup_batches,
    ), batch_rows



@torch.no_grad()
def eval_main(cfg: Config, Ds, infer_module, data_loader=None, dataset_name="MNIST"):
    # 推論のみ
    infer_module.eval()

    _, data_loader = resolve_eval_loader(cfg, data_loader=data_loader, dataset_name=dataset_name)
    Ds_for_inference, lambdas, betas, etas_eval = build_eval_inference_context(cfg, Ds)

    total_loss = 0.0
    total_time_ms = 0.0
    total_samples = 0

    for img, _ in data_loader:
        x = prepare_eval_batch(cfg, img)

        sync_if_needed(cfg.device)
        t0 = time.time()

        codes = infer_module(
            x=x,
            Ds_for_inference=Ds_for_inference,
            lambdas=lambdas,
            betas=betas,
            infer_steps=cfg.infer_steps,
            eta_scale=cfg.eta_scale,
            etas=etas_eval,   # 事前計算済みを使う
        )

        sync_if_needed(cfg.device)
        dt_ms = 1000.0 * (time.time() - t0)

        loss, rec_x, rec_h, sparse = hierarchical_losses(
            x=x,
            Ds=list(Ds),
            codes=codes,
            lambdas=lambdas,
            betas=betas,
        )

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_time_ms += dt_ms
        total_samples += bs

    return {
        "avg_loss": total_loss / total_samples,
        "avg_ms_per_batch": total_time_ms / len(data_loader),
        "avg_ms_per_sample": total_time_ms / total_samples,
    }


if __name__ == "__main__":
    cfg = Config()
    train_main(cfg)
