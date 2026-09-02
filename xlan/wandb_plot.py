#!/usr/bin/env python3
"""Plot named train/validation comparisons from W&B metric histories.

The module supports four data sources:

* a local W&B run directory (``wandb/run-...``) or ``run-*.wandb`` file;
* an online run written as ``wandb://entity/project/run_id``;
* an exported ``.json``, ``.jsonl``/``.ndjson``, or ``.csv`` history file;
* a run's ``files/wandb-summary.json`` (one final point only).

The default ``teacher`` plot and the ``dynamic`` plot share one implementation::

    python xlan/wandb_plot.py
    python xlan/wandb_plot.py dynamic

Only local ``.wandb`` parsing needs the ``wandb`` package; plotting additionally
needs matplotlib. CSV/JSON reading and stitching use the standard library only.
"""

from __future__ import annotations

import argparse
import csv
from decimal import MIN_ETINY
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import matplotlib.pyplot as plt  # type: ignore
from matplotlib.ticker import PercentFormatter  # type: ignore


DEFAULT_X_KEY = "training/epoch"
INTERNAL_KEYS = {"_runtime", "_step", "_timestamp", "_wandb"}

# 每个命名配置都共用相同结构：METHODS 定义实验曲线，PANELS 定义指标。
# RUN 支持本地 run、run-*.wandb、CSV/JSON 或 wandb://entity/project/run_id。
COMMON_COMPARISON = {
    "FIGSIZE": [7.0, 3.5],
    "SUBPLOT_ASPECT": 0.6,
    "WSPACE": 0.32,
    "MARGINS": {"left": 0.10, "right": 0.98, "bottom": 0.23, "top": 0.90},
    "LEGEND": {
        "loc": "lower center",
        "bbox_to_anchor": [0.5, 0.10],
        "ncol": 2,
    },
    "X": "_step",
    "X_LABEL": "Epoch",
    "Y_LIM": [0, 1],
    "PERCENT_Y": True,
    "SHOW_RAW": False,
    "PANELS": [
        {
            "METRIC": "episode/success_rate",
            "TITLE": "(a) Train success rate",
            "Y_LABEL": "Train success rate",
            "SMOOTH": {"method": "ema", "alpha": 0.4},
        },
        {
            "METRIC": "val/success_rate",
            "TITLE": "(b) Val success rate",
            "Y_LABEL": "Val success rate",
            # Validation is sparse (every 5 epochs); plot the measured values.
            "SMOOTH": None,
        },
    ],
}

teacher_plot = {
    **COMMON_COMPARISON,
    "OUTPUTS": ["xlan/figures/GiGPO_teacher_success_rate.pdf"],
    "METHODS": [
        {
            "LABEL": "teacher coef=1.0",
            "COLOR": "#0072B2",
            "STD_ALPHA": 0.16,
            "RUN_GROUPS": [
                [
                    {"RUN": "wandb/run-20260828_102841-78o6d5i8", "MIN_X": 0},
                ],
                [
                    {"RUN": "wandb/run-20260824_013138-78b3leng", "MIN_X": 0, "MAX_X": 45},
                    {"RUN": "wandb/run-20260824_140918-rslak2vu", "MIN_X": 45},
                ],
            ],
        },
        {
            "LABEL": "teacher coef=0.0",
            "COLOR": "#D55E00",
            "STD_ALPHA": 0.16,
            "RUN_GROUPS": [
                [
                    {"RUN": "wandb/run-20260824_225603-ldd5mnbh", "MIN_X": 0},
                ],
                [
                    {"RUN": "wandb/run-20260827_202832-bijsrztf", "MIN_X": 0},
                ],
            ],
        },
    ],
}

dynamic_plot = {
    **COMMON_COMPARISON,
    "OUTPUTS": ["xlan/figures/GiGPO_dynamic_success_rate.pdf"],
    "METHODS": [
        {
            "LABEL": "dapo only",
            "COLOR": "#0072B2",
            "RUNS": [
                {"RUN": "wandb/run-20260810_173335-kg7q4isl", "MIN_X": 0, "MAX_X": 50},
            ],
        },
        {
            "LABEL": "with dynamic",
            "COLOR": "#D55E00",
            "RUNS": [
                {"RUN": "wandb/run-20260807_194646-ft83r4ih", "MIN_X": 0, "MAX_X": 50},
            ],
        },
    ],
}

PLOT_CONFIGS: dict[str, Mapping[str, Any]] = {
    "teacher": teacher_plot,
    "dynamic": dynamic_plot,
}

# Bayesian sampler mechanism figure. It visualizes the actual sampling rule:
# per-seed Jeffreys posterior means q_s and their normalized probability mass.
# Run: python xlan/wandb_plot.py bayes-sampler
BAYES_SAMPLER_PLOT = {
    "OUTPUTS": [
        "xlan/figures/GiGPO_bayesian_seed_sampler.pdf",
        "xlan/figures/GiGPO_bayesian_seed_sampler.png",
    ],
    "TITLE": "",
    "X_LABEL": "Epoch",
    "POSTERIOR_Y_LABEL": "Mean posterior usefulness",
    "PROBABILITY_Y_LABEL": "Sampling probability mass",
    "FIGSIZE": COMMON_COMPARISON["FIGSIZE"],
    "SUBPLOT_ASPECT": COMMON_COMPARISON["SUBPLOT_ASPECT"],
    "WSPACE": COMMON_COMPARISON["WSPACE"],
    "MARGINS": COMMON_COMPARISON["MARGINS"],
    "LEGEND": {
        **COMMON_COMPARISON["LEGEND"],
        "bbox_to_anchor": [0.5, 0.13],
    },
    "STD_ALPHA": 0.14,
    "SUCCESS_CURVE_LABEL": "with dynamic",
    
    # y-axis config
    "POSTERIOR_TITLE": "(a) Posterior usefulness by difficulty",
    "PROBABILITY_TITLE": "(b) Sampling mass by difficulty",
    "PERCENT_Y": True,
    "POSTERIOR_Y_LIM": [0.7, 1],  # posterior usefulness
    "PROBABILITY_Y_LIM": None,
    "SHOW_UNIFORM_REFERENCE": True,
    "GROUPS": [
        {"LABEL": "Easy", "SEEDS": [3, 2, 0, 1], "COLOR": "#009E73"},
        {"LABEL": "Hard", "SEEDS": [4, 5, 6, 7], "COLOR": "#D55E00"},
    ],
    
    # Three complete 50-epoch runs with the same Bayesian sampler settings
    "CURVES": [
        {
            "LABEL": "with dynamic",
            "RUN_GROUPS": [
                [
                    {"RUN": "wandb/run-20260807_194646-ft83r4ih", "MIN_X": 0, "MAX_X": 49},
                ],
                [
                    {"RUN": "wandb/run-20260816_180214-xt0pphsv", "MIN_X": 0, "MAX_X": 49},
                ],
                [
                    {"RUN": "wandb/run-20260817_090742-zpw4p3gx", "MIN_X": 0, "MAX_X": 49},
                ],
            ],
        },
    ],
}



# Aggregate the four chemistry difficulty evaluations (N=1..4) for each
# training setup. ``withdynamic`` is retained as an alias for result files
# produced by the current evaluation jobs.
EVAL_BAR_PLOT = {
    "OUTPUTS": [
        "xlan/figures/GiGPO_eval_metrics.pdf",
        "xlan/figures/GiGPO_eval_metrics.png",
    ],
    "FIGSIZE": [7.0, 3.5],
    "Y_LIM": [0, 1],
    "GROUPS": [
        {
            "LABEL": "nodynamic\nteacher=1.0",
            "PREFIXES": ["eval-nodynamic-teacher1.0"],
        },
        {
            "LABEL": "dynamic\nteacher=1.0",
            "PREFIXES": ["eval-dynamic-teacher1.0", "eval-withdynamic-teacher1.0"],
        },
        {
            "LABEL": "dynamic\nteacher=0.0",
            "PREFIXES": ["eval-dynamic-teacher0.0", "eval-withdynamic-teacher0.0"],
        },
        {
            "LABEL": "nodynamic\nteacher=0.0",
            "PREFIXES": ["eval-nodynamic-teacher0.0"],
        },
    ],
    "METRICS": [
        {"LABEL": "Success rate", "COLOR": "#0072B2"},
        {"LABEL": "Valid action ratio", "COLOR": "#D55E00"},
        {"LABEL": "Task efficiency", "COLOR": "#009E73"},
    ],
    "LEGEND": {
        "loc": "lower center",
        "bbox_to_anchor": [0.5, 0.02],
        "ncol": 3,
    },
    "MARGINS": {"left": 0.10, "right": 0.98, "bottom": 0.30, "top": 0.95},
}


class WandbDataError(RuntimeError):
    """A user-facing input, dependency, or W&B parsing error."""


@dataclass(frozen=True)
class Segment:
    """One run slice used to construct a continuous curve.

    Bounds apply before ``x_shift`` and are inclusive. Set ``auto_continue`` to
    align the segment's first point one inferred x-step after the previous
    segment. ``x_shift`` and ``auto_continue`` are mutually exclusive.
    """

    run: str | Path
    min_x: float | None = None
    max_x: float | None = None
    min_y: float | None = None
    max_y: float | None = None
    x_shift: float = 0.0
    auto_continue: bool = False


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, str):
        try:
            result = float(value)
        except ValueError:
            return None
    else:
        return None
    return result if math.isfinite(result) else None


def _json_value(value: Any) -> Any:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def _resolve_run_file(source: Path) -> Path | None:
    if source.is_file() and source.suffix == ".wandb":
        return source
    if source.is_dir():
        files = sorted(source.glob("run-*.wandb"))
        if len(files) > 1:
            raise WandbDataError(f"More than one .wandb file found in {source}")
        return files[0] if files else None
    return None


def _history_item_key(item: Any) -> str:
    key = getattr(item, "key", "")
    if key:
        return str(key)
    nested = getattr(item, "nested_key", ())
    try:
        return "/".join(str(part) for part in nested)
    except TypeError:
        return str(nested) if nested else ""


def _read_local_wandb(path: Path) -> Iterator[dict[str, Any]]:
    try:
        from wandb.proto import wandb_internal_pb2  # type: ignore
        from wandb.sdk.internal.datastore import DataStore  # type: ignore
    except (ImportError, ModuleNotFoundError) as exc:
        raise WandbDataError(
            "Reading run-*.wandb history requires the W&B SDK. Install the "
            "plotting dependencies with: pip install 'wandb>=0.19' matplotlib"
        ) from exc

    store = DataStore()
    try:
        store.open_for_scan(str(path))
        while True:
            raw = store.scan_data()
            if raw is None:
                break
            record = wandb_internal_pb2.Record()
            record.ParseFromString(raw)
            if record.WhichOneof("record_type") != "history":
                continue
            row: dict[str, Any] = {}
            for item in record.history.item:
                key = _history_item_key(item)
                if key:
                    row[key] = _json_value(item.value_json)
            if row:
                yield row
    except WandbDataError:
        raise
    except Exception as exc:
        raise WandbDataError(f"Could not parse local W&B file {path}: {exc}") from exc
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()


def _read_online_run(source: str, keys: Sequence[str] | None) -> Iterator[dict[str, Any]]:
    try:
        import wandb  # type: ignore
    except (ImportError, ModuleNotFoundError) as exc:
        raise WandbDataError(
            "Online runs require the W&B SDK: pip install 'wandb>=0.19'"
        ) from exc
    run_path = source.removeprefix("wandb://").strip("/")
    if len(run_path.split("/")) != 3:
        raise WandbDataError(
            "Online source must be wandb://entity/project/run_id"
        )
    try:
        run = wandb.Api().run(run_path)
        # scan_history(keys=...) intentionally returns rows containing every
        # requested key, which is exactly what metric extraction needs.
        yield from run.scan_history(keys=list(keys) if keys else None)
    except Exception as exc:
        raise WandbDataError(f"Could not read online W&B run {run_path}: {exc}") from exc


def _read_csv(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            yield {key: _json_value(value) for key, value in row.items()}


def _read_json(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, Mapping):
        payload = payload.get("history", [payload])
    if not isinstance(payload, list):
        raise WandbDataError(f"Expected a JSON row/list or {{'history': [...]}} in {path}")
    for row in payload:
        if isinstance(row, Mapping):
            yield dict(row)


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise WandbDataError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
            if isinstance(row, Mapping):
                yield dict(row)


def iter_history(
    source: str | Path, keys: Sequence[str] | None = None
) -> Iterator[dict[str, Any]]:
    """Iterate raw history rows from any supported source."""

    source_text = str(source)
    if source_text.startswith("wandb://"):
        yield from _read_online_run(source_text, keys)
        return

    path = Path(source).expanduser()
    if not path.exists():
        raise WandbDataError(f"Data source does not exist: {path}")
    run_file = _resolve_run_file(path)
    if run_file:
        yield from _read_local_wandb(run_file)
        return
    if path.is_dir():
        summary = path / "files" / "wandb-summary.json"
        if summary.exists():
            yield from _read_json(summary)
            return
        raise WandbDataError(f"No run-*.wandb or files/wandb-summary.json found in {path}")
    if path.suffix.lower() == ".csv":
        yield from _read_csv(path)
    elif path.suffix.lower() in {".jsonl", ".ndjson"}:
        yield from _read_jsonl(path)
    elif path.suffix.lower() == ".json":
        yield from _read_json(path)
    else:
        raise WandbDataError(f"Unsupported data source: {path}")


def list_fields(source: str | Path) -> list[str]:
    """Return all keys present in a run/history source."""

    fields: set[str] = set()
    for row in iter_history(source):
        fields.update(row)
    return sorted(fields, key=lambda key: (key in INTERNAL_KEYS, key))


def read_metric(
    source: str | Path,
    metric: str,
    x_key: str = DEFAULT_X_KEY,
    *,
    min_x: float | None = None,
    max_x: float | None = None,
) -> list[dict[str, Any]]:
    """Read numeric ``x_key``/``metric`` points, sorted by x.

    Rows missing either requested key and non-finite numeric values are skipped.
    Repeated x values are retained; ``stitch_runs`` controls overlap handling.
    """

    result: list[dict[str, Any]] = []
    
    for row in iter_history(source, keys=[x_key, metric]):
        x_value = _number(row.get(x_key))
        y_value = _number(row.get(metric))
        
        if x_value is None or y_value is None:
            continue
        if min_x is not None and x_value < min_x:
            continue
        if max_x is not None and x_value > max_x:
            continue
        
        result.append({x_key: x_value, metric: y_value})
        
    result.sort(key=lambda row: row[x_key])
    
    if not result:
        available = list_fields(source)
        preview = ", ".join(available[:20]) or "(none)"
        raise WandbDataError(
            f"No numeric points found for x={x_key!r}, metric={metric!r}. "
            f"Available fields include: {preview}"
        )
    return result


def _inferred_step(points: Sequence[Mapping[str, Any]], x_key: str) -> float:
    differences = [
        float(right[x_key]) - float(left[x_key])
        for left, right in zip(points, points[1:])
        if float(right[x_key]) > float(left[x_key])
    ]
    return statistics.median(differences) if differences else 1.0


def stitch_runs(
    segments: Sequence[Segment],
    metric: str,
    x_key: str = DEFAULT_X_KEY,
    *,
    overlap: str = "last",
) -> list[dict[str, Any]]:
    """Slice, shift, and concatenate run histories into one curve.

    ``overlap`` is one of ``first``, ``last``, ``mean``, or ``error`` and
    controls points sharing an x value after shifting.
    """

    if not segments:
        raise WandbDataError("At least one segment is required")
    if overlap not in {"first", "last", "mean", "error"}:
        raise WandbDataError(f"Unknown overlap policy: {overlap}")

    all_points: list[dict[str, Any]] = []
    previous_end: float | None = None
    previous_step = 1.0
    for segment_index, segment in enumerate(segments):
        if segment.auto_continue and segment.x_shift:
            raise WandbDataError("A segment cannot set both x_shift and auto_continue")
        points = read_metric(
            segment.run,
            metric,
            x_key,
            min_x=segment.min_x,
            max_x=segment.max_x,
        )
        shift = float(segment.x_shift)
        if segment.auto_continue and previous_end is not None:
            shift = previous_end + previous_step - float(points[0][x_key])
        shifted = [
            {
                x_key: float(point[x_key]) + shift,
                metric: float(point[metric]),
                "_source": str(segment.run),
                "_segment": segment_index,
            }
            for point in points
        ]
        previous_step = _inferred_step(shifted, x_key)
        previous_end = float(shifted[-1][x_key])
        all_points.extend(shifted)

    grouped: dict[float, list[dict[str, Any]]] = {}
    
    for point in all_points:
        grouped.setdefault(float(point[x_key]), []).append(point)
        
    stitched: list[dict[str, Any]] = []
    
    for x_value in sorted(grouped):
        candidates = grouped[x_value]
        
        if len(candidates) == 1 or overlap == "first":
            chosen = dict(candidates[0])
        elif overlap == "last":
            chosen = dict(candidates[-1])
        elif overlap == "mean":
            chosen = dict(candidates[-1])
            chosen[metric] = statistics.fmean(float(row[metric]) for row in candidates)
            chosen["_source"] = "+".join(str(row["_source"]) for row in candidates)
        else:
            sources = ", ".join(str(row["_source"]) for row in candidates)
            raise WandbDataError(f"Overlapping x={x_value:g} from: {sources}")
        
        stitched.append(chosen)
        
    return stitched


def smooth_values(values: Sequence[float], spec: Mapping[str, Any] | None) -> list[float]:
    """Apply an EMA or centered moving-average smoothing specification."""

    if not spec or spec.get("method", "none") == "none":
        return list(values)
    method = str(spec.get("method"))
    
    if method == "ema":
        alpha = float(spec.get("alpha", 0.2))
        if not 0 < alpha <= 1:
            raise WandbDataError("EMA alpha must be in (0, 1]")
        output: list[float] = []
        
        for value in values:
            output.append(value if not output else alpha * value + (1 - alpha) * output[-1])
        return output
    
    if method == "moving_average":
        window = int(spec.get("window", 5))
        if window < 1:
            raise WandbDataError("Moving-average window must be >= 1")
        
        radius = window // 2
        
        return [
            statistics.fmean(values[max(0, i - radius) : min(len(values), i + radius + 1)])
            for i in range(len(values))
        ]
        
    raise WandbDataError(f"Unknown smoothing method: {method}")


def _run_segments(raw_runs: Sequence[Mapping[str, Any]], project_root: Path) -> list[Segment]:
    segments: list[Segment] = []
    
    for raw_run in raw_runs:
        run = str(raw_run["RUN"])
        
        if not run.startswith("wandb://"):
            candidate = Path(run).expanduser()
            if not candidate.is_absolute():
                run = str(project_root / candidate)
                
        segments.append(
            Segment(
                run=run,
                min_x=_number(raw_run.get("MIN_X")),
                max_x=_number(raw_run.get("MAX_X")),
                x_shift=float(raw_run.get("X_SHIFT", 0.0)),
                auto_continue=bool(raw_run.get("AUTO_CONTINUE", False)),
            )
        )
    return segments


def _method_run_groups(method: Mapping[str, Any]) -> list[list[Mapping[str, Any]]]:
    raw_groups = method.get("RUN_GROUPS")
    
    if raw_groups is None:
        raw_runs = method.get("RUNS", [])
        if not isinstance(raw_runs, list) or not raw_runs:
            raise WandbDataError(f"Method {method.get('LABEL')!r} needs RUNS")
        return [raw_runs]
    
    if not isinstance(raw_groups, list) or not raw_groups:
        raise WandbDataError(f"Method {method.get('LABEL')!r} needs RUN_GROUPS")
    
    groups: list[list[Mapping[str, Any]]] = []
    
    for raw_group in raw_groups:
        if not isinstance(raw_group, list) or not raw_group:
            raise WandbDataError(
                f"Every RUN_GROUPS entry for {method.get('LABEL')!r} must be non-empty"
            )
        groups.append(raw_group)
        
    return groups


def _mean_std_band(
    curves: Sequence[tuple[Sequence[float], Sequence[float]]],
) -> tuple[list[float], list[float], list[float], list[float]]:
    
    values_by_x: dict[float, list[float]] = {}
    
    for xs, ys in curves:
        for x_value, y_value in zip(xs, ys):
            values_by_x.setdefault(float(x_value), []).append(float(y_value))
            
    xs = sorted(values_by_x)
    
    means: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    for x_value in xs:
        values = values_by_x[x_value]
        mean = statistics.fmean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        means.append(mean)
        lower.append(mean - std)
        upper.append(mean + std)
        
    return xs, means, lower, upper


PAPER_STYLE: dict[str, Any] = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.55,
    "legend.fontsize": 8,
    "lines.linewidth": 1.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
}


def plot_comparison(settings: Mapping[str, Any]) -> Path | list[Path]:
    """Render a named train/val comparison with one shared legend."""

    panels = settings.get("PANELS", [])
    if not isinstance(panels, list) or not panels:
        raise WandbDataError("Plot configuration needs a non-empty PANELS list")
    
    methods = settings.get("METHODS", [])
    if not isinstance(methods, list) or not methods:
        raise WandbDataError("Plot configuration needs a non-empty METHODS list")

    project_root = Path(__file__).resolve().parent.parent
    with plt.rc_context(dict(PAPER_STYLE)):
        fig, axes = plt.subplots(
            1,
            len(panels),
            figsize=settings.get("FIGSIZE", [3.5 * len(panels), 3.5]),
            squeeze=False,
        )
        flat_axes = list(axes[0])
        aspect = float(settings.get("SUBPLOT_ASPECT", 1.0))

        for ax, panel in zip(flat_axes, panels):
            if not isinstance(panel, Mapping) or "METRIC" not in panel:
                raise WandbDataError("Every panel needs a METRIC")
            
            metric = str(panel["METRIC"])
            x_key = str(panel.get("X", settings.get("X", DEFAULT_X_KEY)))
            
            for method in methods:
                if not isinstance(method, Mapping):
                    raise WandbDataError("Every method needs to be a mapping")
                
                smooth = method.get("SMOOTH", panel.get("SMOOTH", settings.get("SMOOTH")))
                curves: list[tuple[list[float], list[float], list[float]]] = []
                
                for raw_runs in _method_run_groups(method):
                    points = stitch_runs(
                        _run_segments(raw_runs, project_root),
                        metric,
                        x_key,
                        overlap=str(method.get("OVERLAP", "last")),
                    )
                    xs = [float(row[x_key]) for row in points]
                    raw_ys = [float(row[metric]) for row in points]
                    ys = smooth_values(raw_ys, smooth)
                    curves.append((xs, raw_ys, ys))
                    
                plot_kwargs = {
                    key.lower(): method[key]
                    for key in ("COLOR", "LINESTYLE", "LINEWIDTH", "MARKER", "MARKERSIZE", "ALPHA")
                    if method.get(key) is not None
                }
                
                show_raw = method.get("SHOW_RAW", panel.get("SHOW_RAW", settings.get("SHOW_RAW", False)))
                if len(curves) == 1:
                    xs, raw_ys, ys = curves[0]
                    if show_raw and ys != raw_ys:
                        raw_kwargs = dict(plot_kwargs)
                        raw_kwargs.update(alpha=0.18, linewidth=0.8)
                        ax.plot(xs, raw_ys, **raw_kwargs)
                    ax.plot(xs, ys, label=str(method.get("LABEL", metric)), **plot_kwargs)
                    continue

                if method.get("SHOW_MEMBERS", False):
                    member_kwargs = dict(plot_kwargs)
                    member_kwargs.update(alpha=float(method.get("MEMBER_ALPHA", 0.22)), linewidth=0.8)
                    for xs, _raw_ys, ys in curves:
                        ax.plot(xs, ys, **member_kwargs)
                        
                mean_xs, means, lower, upper = _mean_std_band(
                    [(xs, ys) for xs, _raw_ys, ys in curves]
                )
                
                if method.get("SHOW_STD", True):
                    ax.fill_between(
                        mean_xs,
                        lower,
                        upper,
                        color=str(method.get("COLOR", "#0072B2")),
                        alpha=float(method.get("STD_ALPHA", 0.14)),
                        linewidth=0,
                    )
                    
                ax.plot(mean_xs, means, label=str(method.get("LABEL", metric)), **plot_kwargs)

            ax.set_box_aspect(aspect)
            ax.set_xlabel(str(panel.get("X_LABEL", settings.get("X_LABEL", x_key))))
            ax.set_ylabel(str(panel.get("Y_LABEL", metric)))
            ax.set_title(str(panel.get("TITLE", "")))
            xlim = panel.get("X_LIM", settings.get("X_LIM"))
            ylim = panel.get("Y_LIM", settings.get("Y_LIM"))
            if xlim is not None:
                ax.set_xlim(*xlim)
            if ylim is not None:
                ax.set_ylim(*ylim)
            if panel.get("PERCENT_Y", settings.get("PERCENT_Y", False)):
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

        legend_entries: dict[str, Any] = {}
        for ax in flat_axes:
            handles, labels = ax.get_legend_handles_labels()
            legend_entries.update(zip(labels, handles))
            
        legend_options = settings.get("LEGEND", {})
        if legend_entries and legend_options is not False:
            options = dict(legend_options) if isinstance(legend_options, Mapping) else {}
            fig.legend(
                list(legend_entries.values()),
                list(legend_entries),
                frameon=False,
                **options,
            )

        margins = settings.get("MARGINS", {})
        adjust_options = dict(margins) if isinstance(margins, Mapping) else {}
        adjust_options["wspace"] = float(settings.get("WSPACE", 0.32))
        fig.subplots_adjust(**adjust_options)

        raw_outputs = settings.get("OUTPUTS", ["xlan/figures/comparison.pdf"])
        if isinstance(raw_outputs, (str, Path)):
            raw_outputs = [raw_outputs]
            
        outputs: list[Path] = []
        for raw_output in raw_outputs:
            output = Path(raw_output).expanduser()
            if not output.is_absolute():
                output = project_root / output
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, dpi=int(settings.get("DPI", 300)))
            outputs.append(output)
        plt.close(fig)
        
    return outputs[0] if len(outputs) == 1 else outputs


_EPOCH_RE = re.compile(r"training/epoch:([-+0-9.eE]+)")
_SEED_SUCCESS_RE = re.compile(
    r"seed_sampler/success_ema/(\d+):([-+0-9.eE]+)"
)
_SEED_PROBABILITY_RE = re.compile(
    r"seed_sampler/probability/(\d+):([-+0-9.eE]+)"
)
_SEED_POSTERIOR_RE = re.compile(
    r"seed_sampler/posterior_mean/(\d+):([-+0-9.eE]+)"
)
_SEED_GROUPS_RE = re.compile(
    r"seed_sampler/groups/(\d+):([-+0-9.eE]+)"
)
_SEED_ACCEPT_RATE_RE = re.compile(
    r"seed_sampler/accept_rate/(\d+):([-+0-9.eE]+)"
)


def _read_seed_success_ema(run: Path) -> dict[float, dict[int, float]]:
    """Read the sampler's per-seed success EMA values from output.log."""

    output_log = run / "files" / "output.log"
    if not output_log.exists():
        return {}
    values: dict[float, dict[int, float]] = {}
    
    with output_log.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            epoch_match = _EPOCH_RE.search(line)
            success_matches = _SEED_SUCCESS_RE.findall(line)
            
            if epoch_match and success_matches:
                epoch = float(epoch_match.group(1))
                values[epoch] = {
                    int(seed): float(success)
                    for seed, success in success_matches
                }
    return values


def _read_seed_probability(run: Path) -> dict[float, dict[int, float]]:
    """Read the sampler's instantaneous per-seed probabilities from output.log."""

    output_log = run / "files" / "output.log"
    if not output_log.exists():
        return {}
    values: dict[float, dict[int, float]] = {}
    with output_log.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            epoch_match = _EPOCH_RE.search(line)
            probability_matches = _SEED_PROBABILITY_RE.findall(line)
            
            if epoch_match and probability_matches:
                epoch = float(epoch_match.group(1))
                values[epoch] = {
                    int(seed): float(probability)
                    for seed, probability in probability_matches
                }
    return values


def _read_seed_posterior_mean(run: Path) -> dict[float, dict[int, float]]:
    """Read per-seed posterior means, reconstructing them for older logs."""

    output_log = run / "files" / "output.log"
    if not output_log.exists():
        return {}
    direct_values: dict[float, dict[int, float]] = {}
    reconstructed: dict[float, dict[int, float]] = {}
    with output_log.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            epoch_match = _EPOCH_RE.search(line)
            if not epoch_match:
                continue
            epoch = float(epoch_match.group(1))
            posterior_matches = _SEED_POSTERIOR_RE.findall(line)
            if posterior_matches:
                direct_values[epoch] = {
                    int(seed): float(value)
                    for seed, value in posterior_matches
                }
                continue
            group_matches = _SEED_GROUPS_RE.findall(line)
            accept_rate_matches = _SEED_ACCEPT_RATE_RE.findall(line)
            if group_matches and accept_rate_matches:
                groups_by_seed = {
                    int(seed): float(groups)
                    for seed, groups in group_matches
                }
                accept_rate_by_seed = {
                    int(seed): float(rate)
                    for seed, rate in accept_rate_matches
                }
                epoch_values: dict[int, float] = {}
                for seed, groups in groups_by_seed.items():
                    if seed not in accept_rate_by_seed or groups < 0:
                        continue
                    accepted = accept_rate_by_seed[seed] * groups
                    rejected = groups - accepted
                    epoch_values[seed] = (accepted + 0.5) / (accepted + rejected + 1.0)
                if epoch_values:
                    reconstructed[epoch] = epoch_values
    return direct_values or reconstructed


def _seed_values_for_curve(
    curve: Mapping[str, Any],
    project_root: Path,
    reader: Callable[[Path], dict[float, dict[int, float]]],
) -> dict[float, dict[int, float]]:
    """Slice and stitch one family of per-seed values across run segments."""

    runs = curve.get("RUNS", [])
    if not isinstance(runs, list) or not runs:
        raise WandbDataError(f"Curve {curve.get('LABEL')!r} needs at least one RUN")
    stitched: dict[float, dict[int, float]] = {}
    previous_end: float | None = None
    previous_step = 1.0
    for raw in runs:
        
        run_path = Path(str(raw["RUN"])).expanduser()
        if not run_path.is_absolute():
            run_path = project_root / run_path
        segment = reader(run_path)
        if not segment:
            continue
        
        min_x = _number(raw.get("MIN_X"))
        max_x = _number(raw.get("MAX_X"))
        local_xs = [
            x
            for x in sorted(segment)
            if (min_x is None or x >= min_x) and (max_x is None or x <= max_x)
        ]
        if not local_xs:
            continue
        
        shift = float(raw.get("X_SHIFT", 0.0))
        if raw.get("AUTO_CONTINUE", False):
            if shift:
                raise WandbDataError("A segment cannot set both X_SHIFT and AUTO_CONTINUE")
            if previous_end is not None:
                shift = previous_end + previous_step - local_xs[0]
        shifted_xs = [x + shift for x in local_xs]
        differences = [
            right - left
            for left, right in zip(shifted_xs, shifted_xs[1:])
            if right > left
        ] # differences: compute the step sizes between consecutive shifted x values
        
        previous_step = statistics.median(differences) if differences else previous_step
        previous_end = shifted_xs[-1]
        
        for local_x, shifted_x in zip(local_xs, shifted_xs):
            stitched[shifted_x] = dict(segment[local_x])
            
    return dict(sorted(stitched.items()))


def bayesian_sampler_plot(
    settings: Mapping[str, Any] = BAYES_SAMPLER_PLOT,
) -> Path | list[Path]:
    """Plot mean Bayesian priority and sampling mass across selected runs."""

    curves = settings.get("CURVES", [])
    groups = settings.get("GROUPS", [])
    if not isinstance(curves, list) or not curves:
        raise WandbDataError("BAYES_SAMPLER_PLOT['CURVES'] must be a non-empty list")
    if not isinstance(groups, list) or not groups:
        raise WandbDataError("BAYES_SAMPLER_PLOT['GROUPS'] must be a non-empty list")
    curve_label = str(settings.get("SUCCESS_CURVE_LABEL", "with dynamic"))
    dynamic_curve = next(
        (curve for curve in curves if str(curve.get("LABEL", "")) == curve_label),
        None,
    )
    if dynamic_curve is None:
        raise WandbDataError(f"No curve labelled {curve_label!r} for the Bayesian sampler plot")

    project_root = Path(__file__).resolve().parent.parent
    posterior_runs: list[dict[float, dict[int, float]]] = []
    probability_runs: list[dict[float, dict[int, float]]] = []
    for run_index, run_group in enumerate(_method_run_groups(dynamic_curve), start=1):
        run_curve = {**dynamic_curve, "RUNS": run_group}
        posterior_by_epoch = _seed_values_for_curve(
            run_curve, project_root, _read_seed_posterior_mean
        )
        probability_by_epoch = _seed_values_for_curve(
            run_curve, project_root, _read_seed_probability
        )
        if not posterior_by_epoch or not probability_by_epoch:
            raise WandbDataError(
                f"Curve {curve_label!r} run group {run_index} needs "
                "seed posterior/probability metrics"
            )
        posterior_runs.append(posterior_by_epoch)
        probability_runs.append(probability_by_epoch)

    observed_seed_sets = [
        {
            seed
            for values_by_epoch in (posterior_by_epoch, probability_by_epoch)
            for values in values_by_epoch.values()
            for seed in values
        }
        for posterior_by_epoch, probability_by_epoch in zip(
            posterior_runs, probability_runs
        )
    ]
    observed_seeds = observed_seed_sets[0]
    if any(seeds != observed_seeds for seeds in observed_seed_sets[1:]):
        raise WandbDataError("Bayesian sampler runs must use the same seed pool")

    configured_seeds: list[int] = []
    for group in groups:
        if not isinstance(group, Mapping):
            raise WandbDataError("Every difficulty group must be a mapping")
        seeds = [int(seed) for seed in group.get("SEEDS", [])]
        if not seeds:
            raise WandbDataError(f"Group {group.get('LABEL')!r} has no seeds")
        configured_seeds.extend(seeds)
        
    if len(configured_seeds) != len(set(configured_seeds)):
        raise WandbDataError("A seed cannot appear in more than one difficulty group")
    
    missing = observed_seeds - set(configured_seeds)
    unknown = set(configured_seeds) - observed_seeds
    if missing or unknown:
        raise WandbDataError(
            "Difficulty groups must cover the observed seed pool exactly; "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )

    def aggregate_group_runs(
        runs: Sequence[dict[float, dict[int, float]]],
        seeds: Sequence[int],
        aggregate: Callable[[list[float]], float],
    ) -> tuple[list[float], list[float], list[float], list[float]]:
        member_curves: list[tuple[list[float], list[float]]] = []
        for values_by_epoch in runs:
            xs: list[float] = []
            ys: list[float] = []
            for epoch in sorted(values_by_epoch):
                values = values_by_epoch[epoch]
                if all(seed in values for seed in seeds):
                    xs.append(epoch)
                    ys.append(aggregate([values[seed] for seed in seeds]))
            member_curves.append((xs, ys))
        return _mean_std_band(member_curves)

    with plt.rc_context(dict(PAPER_STYLE)):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=settings.get("FIGSIZE", COMMON_COMPARISON["FIGSIZE"]),
            sharex=True,
            sharey=False,
        )
        aspect = float(settings.get("SUBPLOT_ASPECT", COMMON_COMPARISON["SUBPLOT_ASPECT"]))
        total_seed_count = len(observed_seeds)
        std_alpha = float(settings.get("STD_ALPHA", 0.14))
        for group in groups:
            label = str(group.get("LABEL", "Group"))
            seeds = [int(seed) for seed in group["SEEDS"]]
            color = str(group.get("COLOR", "#0072B2"))

            xs, means, lower, upper = aggregate_group_runs(
                posterior_runs, seeds, statistics.fmean
            )
            axes[0].fill_between(
                xs,
                [max(0.0, value) for value in lower],
                [min(1.0, value) for value in upper],
                color=color,
                alpha=std_alpha,
                linewidth=0,
            )
            axes[0].plot(xs, means, color=color, label=label)

            xs, means, lower, upper = aggregate_group_runs(
                probability_runs, seeds, sum
            )
            axes[1].fill_between(
                xs,
                [max(0.0, value) for value in lower],
                [min(1.0, value) for value in upper],
                color=color,
                alpha=std_alpha,
                linewidth=0,
            )
            axes[1].plot(xs, means, color=color, label=label)

        if settings.get("SHOW_UNIFORM_REFERENCE", True):
            for group in groups:
                reference = len([int(seed) for seed in group["SEEDS"]]) / total_seed_count
                axes[1].axhline(
                    reference,
                    color="black",
                    linestyle=":",
                    linewidth=0.9,
                    alpha=0.5,
                    zorder=0,
                )

        axes[0].set_title(str(settings.get("POSTERIOR_TITLE", "(a) Posterior usefulness")))
        axes[1].set_title(str(settings.get("PROBABILITY_TITLE", "(b) Sampling mass")))
        axes[0].set_ylabel(str(settings.get("POSTERIOR_Y_LABEL", "Mean posterior usefulness")))
        axes[1].set_ylabel(str(settings.get("PROBABILITY_Y_LABEL", "Sampling probability mass")))
        
        # ax formatting
        for ax in axes:
            ax.set_box_aspect(aspect)
            ax.set_xlabel(str(settings.get("X_LABEL", "Epoch")))
            if settings.get("PERCENT_Y", True):
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        
        # y-axis limit
        if settings.get("POSTERIOR_Y_LIM") is not None:
            axes[0].set_ylim(*settings["POSTERIOR_Y_LIM"])
        
        if settings.get("PROBABILITY_Y_LIM") is not None:
            axes[1].set_ylim(*settings["PROBABILITY_Y_LIM"])
        
        # legend
        handles, labels = axes[1].get_legend_handles_labels()
        legend_options = settings.get("LEGEND", COMMON_COMPARISON["LEGEND"])
        if handles and legend_options is not False:
            options = dict(legend_options) if isinstance(legend_options, Mapping) else {}
            fig.legend(handles, labels, frameon=False, **options)
        
        # title
        if settings.get("TITLE"):
            fig.suptitle(str(settings["TITLE"]))
        
        # layout
        margins = settings.get("MARGINS", COMMON_COMPARISON["MARGINS"])
        adjust_options = dict(margins) if isinstance(margins, Mapping) else {}
        adjust_options["wspace"] = float(settings.get("WSPACE", COMMON_COMPARISON["WSPACE"]))
        fig.subplots_adjust(**adjust_options)
        
        # output path
        raw_outputs = settings.get("OUTPUTS", ["xlan/figures/bayesian_seed_sampler.pdf"])
        if isinstance(raw_outputs, (str, Path)):
            raw_outputs = [raw_outputs]
        outputs: list[Path] = []
        
        for raw_output in raw_outputs:
            output = Path(raw_output).expanduser()
            if not output.is_absolute():
                output = project_root / output
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, dpi=int(settings.get("DPI", 300)))
            outputs.append(output)
            
        plt.close(fig)
        
    return outputs[0] if len(outputs) == 1 else outputs


# Ablation study
# Evaluation metrics across complexity levels N=1..4, grouped by training setup.

def _evaluation_metrics(path: Path, expected_n: int) -> tuple[float, float, float]:
    """Read success, validity, and normalized efficiency from one eval JSON."""

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise WandbDataError(f"Could not read evaluation result {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise WandbDataError(f"Evaluation result must be a JSON object: {path}")

    metrics = payload.get("metrics")
    parameters = payload.get("eval_parameters")
    if not isinstance(metrics, Mapping) or not isinstance(parameters, Mapping):
        raise WandbDataError(f"Missing metrics/eval_parameters in {path}")

    actual_n = _number(parameters.get("max_chemical_n"))
    if actual_n is not None and actual_n != expected_n:
        raise WandbDataError(
            f"Expected N={expected_n} but {path} records N={actual_n:g}"
        )

    def metric_mean(name: str) -> float:
        entry = metrics.get(name)
        value = _number(entry.get("mean")) if isinstance(entry, Mapping) else None
        if value is None:
            raise WandbDataError(f"Missing numeric metrics.{name}.mean in {path}")
        return value

    success_rate = metric_mean("success_rate")
    valid_ratio = metric_mean("valid_ratio")
    episode_length = metric_mean("episode_length")
    max_steps = _number(parameters.get("max_steps"))
    if max_steps is None or max_steps <= 0:
        raise WandbDataError(f"Missing positive eval_parameters.max_steps in {path}")
    task_efficiency = (max_steps - episode_length) / max_steps
    return success_rate, valid_ratio, task_efficiency


def _evaluation_group_files(
    results_dir: Path, group: Mapping[str, Any]
) -> list[Path]:
    prefixes = group.get("PREFIXES", [])
    if not isinstance(prefixes, list) or not prefixes:
        raise WandbDataError(f"Evaluation group {group.get('LABEL')!r} needs PREFIXES")
    for prefix in prefixes:
        files = [results_dir / f"{prefix}-n{n}.json" for n in range(1, 5)]
        if all(path.is_file() for path in files):
            return files
    expected = " or ".join(f"{prefix}-n1..n4.json" for prefix in prefixes)
    raise WandbDataError(
        f"Evaluation group {group.get('LABEL')!r} is incomplete; expected {expected} "
        f"under {results_dir}"
    )


def evaluation_bar_plot(
    settings: Mapping[str, Any] = EVAL_BAR_PLOT,
) -> Path | list[Path]:
    """Plot N=1..4 mean evaluation metrics grouped by training setup."""

    groups = settings.get("GROUPS", [])
    metric_specs = settings.get("METRICS", [])
    if not isinstance(groups, list) or not groups:
        raise WandbDataError("EVAL_BAR_PLOT['GROUPS'] must be a non-empty list")
    if not isinstance(metric_specs, list) or len(metric_specs) != 3:
        raise WandbDataError("EVAL_BAR_PLOT['METRICS'] must define three metrics")

    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "xlan" / "results"
    group_values: list[tuple[float, float, float]] = []
    
    for group in groups:
        if not isinstance(group, Mapping):
            raise WandbDataError("Every evaluation group must be a mapping")
        files = _evaluation_group_files(results_dir, group)
        per_n = [
            _evaluation_metrics(path, n)
            for n, path in enumerate(files, start=1)
        ]
        group_values.append(
            tuple(statistics.fmean(values) for values in zip(*per_n))
        )

    with plt.rc_context(dict(PAPER_STYLE)):
        fig, ax = plt.subplots(figsize=settings.get("FIGSIZE", [7.0, 3.5]))
        centers = list(range(len(groups)))
        width = 0.24
        offsets = (-width, 0.0, width)
        for metric_index, (metric, offset) in enumerate(zip(metric_specs, offsets)):
            ax.bar(
                [center + offset for center in centers],
                [values[metric_index] for values in group_values],
                width=width,
                label=str(metric.get("LABEL", f"Metric {metric_index + 1}")),
                color=str(metric.get("COLOR", "#0072B2")),
            )

        ax.set_xticks(centers, [str(group.get("LABEL", "Group")) for group in groups])
        ax.set_ylabel("Mean score across N=1–4")
        ax.set_ylim(*settings.get("Y_LIM", [0, 1]))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)

        legend_options = settings.get("LEGEND", {})
        options = dict(legend_options) if isinstance(legend_options, Mapping) else {}
        fig.legend(frameon=False, **options)
        margins = settings.get("MARGINS", {})
        fig.subplots_adjust(**(dict(margins) if isinstance(margins, Mapping) else {}))

        raw_outputs = settings.get("OUTPUTS", ["xlan/figures/GiGPO_eval_metrics.pdf"])
        if isinstance(raw_outputs, (str, Path)):
            raw_outputs = [raw_outputs]
        outputs: list[Path] = []
        for raw_output in raw_outputs:
            output = Path(raw_output).expanduser()
            if not output.is_absolute():
                output = project_root / output
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, dpi=int(settings.get("DPI", 300)))
            outputs.append(output)
        plt.close(fig)

    return outputs[0] if len(outputs) == 1 else outputs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot a named W&B experiment comparison (default: teacher)."
    )
    choices = [
        *PLOT_CONFIGS,
        "seed-analysis",
        "seed-group-analysis",
        "bayes-sampler",
        "eval-bars",
    ]
    parser.add_argument(
        "plot",
        nargs="?",
        default="teacher",
        choices=choices,
        help="named plot configuration (default: teacher)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.plot in PLOT_CONFIGS:
        result = plot_comparison(PLOT_CONFIGS[args.plot])
    elif args.plot == "bayes-sampler":
        result = bayesian_sampler_plot()
    elif args.plot == "eval-bars":
        result = evaluation_bar_plot()
        
    for path in result if isinstance(result, list) else [result]:
        print(f"Wrote figure to {path}")
            
    return 0


if __name__ == "__main__":
    main()
