#!/usr/bin/env python3
"""Analyze ssne_ai_demo performance logs and draw latency charts."""

from __future__ import annotations

import argparse
import math
import os
import re
import statistics
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


Number = float
PerfDict = Dict[str, Number]
Record = Dict[str, object]

PERF_RE = re.compile(r"\[PERF\]\[(?P<section>[^\]]+)\]\s*(?P<body>.*)")
KV_RE = re.compile(r"(?P<key>[A-Za-z0-9_]+)=(?P<value>\"[^\"]*\"|\S+)")

TOP_STAGE_ORDER = ("get_image", "palm_total", "hand_total", "draw", "process", "loop")
PALM_DETAIL_ORDER = (
    "palm_preprocess",
    "palm_preprocess_transform",
    "palm_preprocess_manual_load",
    "palm_input_load",
    "palm_inference",
    "palm_getoutput",
    "palm_output_meta",
    "palm_decode",
    "palm_verbose_log",
    "palm_accounted",
)
SUMMARY_KEYS = (
    "app_fps_window",
    "app_fps_total",
    "R",
    "drop_rate_est_pct",
    "realtime_score_est",
)
LATENCY_KEYS = (
    "e2e_loop_p95_ms",
    "e2e_loop_p95_T",
    "e2e_process_p95_ms",
    "e2e_process_p95_T",
    "latency_score_est_by_loop",
)
JITTER_KEYS = (
    "loop_ms_avg",
    "loop_ms_p95",
    "loop_jitter_p95_vs_avg_pct",
    "instant_fps_avg",
    "instant_fps_p95",
    "fps_jitter_p95_vs_avg_pct",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze ssne_ai_demo [PERF] logs and export TXT/SVG statistics."
    )
    parser.add_argument(
        "--mode",
        choices=("palm", "palm_hand"),
        required=True,
        help="Test program mode. Use palm for palm-only logs, palm_hand for cascaded palm + hand logs.",
    )
    parser.add_argument(
        "--kInferInterval",
        type=positive_int,
        required=True,
        help="Inference interval used by ssne_ai_demo. 1 means infer every frame.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        required=True,
        help="Path to result.log or another ssne_ai_demo log file.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output directory for TXT and SVG files. Defaults to the log file directory.",
    )
    return parser.parse_args()


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def parse_value(raw: str) -> object:
    if raw.startswith('"') and raw.endswith('"'):
        return raw[1:-1]
    try:
        if any(ch in raw for ch in ".eE"):
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def parse_kv_body(body: str) -> Dict[str, object]:
    values: Dict[str, object] = {}
    for match in KV_RE.finditer(body):
        values[match.group("key")] = parse_value(match.group("value"))
    return values


def parse_perf_log(log_path: Path) -> Tuple[Dict[str, object], List[Record], List[str]]:
    config: Dict[str, object] = {}
    records: List[Record] = []
    warnings: List[str] = []
    current: Optional[Record] = None

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, start=1):
            match = PERF_RE.search(line)
            if not match:
                continue

            section = match.group("section")
            data = parse_kv_body(match.group("body"))

            if section == "config":
                config.update(data)
                continue

            if section == "summary":
                current = {"summary": data, "line": lineno, "index": len(records) + 1}
                records.append(current)
                continue

            if current is None:
                warnings.append(f"Line {lineno}: [{section}] appears before the first summary and was ignored.")
                continue

            current[section] = data

    return config, records, warnings


def as_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def get_section(record: Record, section: str) -> PerfDict:
    data = record.get(section, {})
    if not isinstance(data, dict):
        return {}
    return {str(k): as_float(v) for k, v in data.items() if isinstance(v, (int, float))}


def value(record: Record, section: str, key: str, default: float = math.nan) -> float:
    data = get_section(record, section)
    if key not in data:
        return default
    return data[key]


def safe_values(records: Sequence[Record], section: str, key: str) -> List[float]:
    vals = [value(record, section, key) for record in records]
    return [v for v in vals if not math.isnan(v)]


def mean_or_nan(vals: Sequence[float]) -> float:
    return statistics.fmean(vals) if vals else math.nan


def percentile(vals: Sequence[float], pct: float) -> float:
    clean = sorted(v for v in vals if not math.isnan(v))
    if not clean:
        return math.nan
    if len(clean) == 1:
        return clean[0]
    idx = (len(clean) - 1) * pct / 100.0
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return clean[int(idx)]
    return clean[lo] + (clean[hi] - clean[lo]) * (idx - lo)


def count_infer_frames(start_frame: int, end_frame: int, k: int) -> int:
    if end_frame < start_frame:
        return 0
    first = ((start_frame + k - 1) // k) * k
    if first > end_frame:
        return 0
    return ((end_frame - first) // k) + 1


def add_inference_counts(records: Sequence[Record], mode: str, k: int) -> None:
    for record in records:
        summary = get_section(record, "summary")
        end_frame = int(summary.get("frame", -1))
        window_frames = int(summary.get("window_frames", 0))
        start_frame = max(0, end_frame - window_frames + 1)
        infer_count = count_infer_frames(start_frame, end_frame, k)
        record["infer_start_frame"] = start_frame
        record["infer_end_frame"] = end_frame
        record["palm_infer_count"] = infer_count
        record["hand_call_count_est"] = infer_count if mode == "palm_hand" else 0


def fmt_num(value_: object, digits: int = 3) -> str:
    if value_ is None:
        return "-"
    if isinstance(value_, int) and not isinstance(value_, bool):
        return str(value_)
    try:
        number = float(value_)
    except (TypeError, ValueError):
        return str(value_)
    if math.isnan(number):
        return "-"
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    if abs(number - round(number)) < 1e-9 and abs(number) >= 10:
        return str(int(round(number)))
    return f"{number:.{digits}f}"


def table(headers: Sequence[str], rows: Sequence[Sequence[object]], digits: int = 3) -> str:
    rendered_rows = [[fmt_num(cell, digits) for cell in row] for row in rows]
    rendered_headers = [str(h) for h in headers]
    widths = [len(h) for h in rendered_headers]
    for row in rendered_rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    def render(row: Sequence[str]) -> str:
        return "  ".join(cell.rjust(width) for cell, width in zip(row, widths))

    sep = "  ".join("-" * width for width in widths)
    lines = [render(rendered_headers), sep]
    lines.extend(render(row) for row in rendered_rows)
    return "\n".join(lines)


def aggregate_rows(records: Sequence[Record], section: str, keys: Iterable[str]) -> List[List[object]]:
    rows: List[List[object]] = []
    for key in keys:
        vals = safe_values(records, section, key)
        rows.append(
            [
                key,
                mean_or_nan(vals),
                percentile(vals, 50.0),
                percentile(vals, 95.0),
                min(vals) if vals else math.nan,
                max(vals) if vals else math.nan,
                vals[-1] if vals else math.nan,
            ]
        )
    return rows


def stage_metric_keys(mode: str) -> List[str]:
    keys: List[str] = []
    for stage in TOP_STAGE_ORDER:
        if mode == "palm" and stage == "hand_total":
            continue
        keys.extend([f"{stage}_avg", f"{stage}_p95"])
    if "loop_max" not in keys:
        keys.append("loop_max")
    return keys


def infer_avg_rows(records: Sequence[Record], mode: str, k: int) -> List[List[object]]:
    if not records:
        return []
    latest = records[-1]
    rows: List[List[object]] = []
    stage = get_section(latest, "stage_ms")
    detail = get_section(latest, "palm_detail_ms")

    for key, label in (
        ("palm_total_avg", "palm_total"),
        ("hand_total_avg", "hand_total"),
    ):
        if key == "hand_total_avg" and mode == "palm":
            continue
        if key in stage:
            rows.append([label, stage[key], stage[key] * k, "latest window avg * k"])

    for name in PALM_DETAIL_ORDER:
        key = f"{name}_avg"
        if key in detail:
            rows.append([name, detail[key], detail[key] * k, "latest window avg * k"])

    return rows


def collect_run_metrics(
    mode: str,
    k: int,
    config: Dict[str, object],
    records: Sequence[Record],
) -> Dict[str, object]:
    if not records:
        return {
            "total_windows": 0,
            "total_reported_frames": 0,
            "last_total_frames": 0,
            "total_palm_infer": 0,
            "total_hand_calls": 0,
            "sensor_fps": as_float(config.get("sensor_fps_cfg", 0.0)),
            "stage_names": [name for name in TOP_STAGE_ORDER if not (mode == "palm" and name == "hand_total")],
            "k": k,
        }

    last_summary = get_section(records[-1], "summary")
    total_reported_frames = sum(int(value(record, "summary", "window_frames", 0.0)) for record in records)
    return {
        "total_windows": len(records),
        "total_reported_frames": total_reported_frames,
        "last_total_frames": int(last_summary.get("total_frames", total_reported_frames)),
        "total_palm_infer": sum(int(record.get("palm_infer_count", 0)) for record in records),
        "total_hand_calls": sum(int(record.get("hand_call_count_est", 0)) for record in records),
        "sensor_fps": last_summary.get("sensor_fps_cfg", as_float(config.get("sensor_fps_cfg", 0.0))),
        "stage_names": [name for name in TOP_STAGE_ORDER if not (mode == "palm" and name == "hand_total")],
        "k": k,
    }


def build_chinese_summary_report(
    *,
    log_path: Path,
    out_dir: Path,
    mode: str,
    k: int,
    config: Dict[str, object],
    records: Sequence[Record],
    parse_warnings: Sequence[str],
    plot_paths: Sequence[Path],
    plot_warnings: Sequence[str],
    detail_txt_path: Path,
    summary_txt_path: Path,
) -> str:
    metrics = collect_run_metrics(mode, k, config, records)
    lines: List[str] = []
    lines.append("ssne_ai_demo 性能摘要 (Performance Summary)")
    lines.append("=" * 46)
    lines.append("")
    lines.append(f"日志文件 (log file): {log_path}")
    lines.append(f"输出目录 (output directory): {out_dir}")
    lines.append(f"详细报告 (detailed report): {detail_txt_path}")
    lines.append("")

    if not records:
        lines.append("未找到性能统计窗口 (performance window)。")
        return "\n".join(lines)

    app_fps_window_vals = safe_values(records, "summary", "app_fps_window")
    app_fps_total_vals = safe_values(records, "summary", "app_fps_total")
    realtime_vals = safe_values(records, "summary", "R")
    drop_vals = safe_values(records, "summary", "drop_rate_est_pct")
    latest_summary = get_section(records[-1], "summary")

    lines.append("核心信息 (Key Metrics)")
    lines.append("-" * 24)
    summary_rows = [
        ["工作模式 (mode)", mode],
        ["推理间隔 (kInferInterval)", k],
        ["统计窗口数 (performance windows)", metrics["total_windows"]],
        ["采集总帧数 (captured frames)", metrics["last_total_frames"]],
        ["统计总帧数 (reported frames)", metrics["total_reported_frames"]],
        ["Palm 推理总帧数 (palm inference frames, est.)", metrics["total_palm_infer"]],
        ["Hand 调用总帧数 (hand call frames, est.)", metrics["total_hand_calls"]],
        ["摄像头配置帧率 (sensor FPS)", metrics["sensor_fps"]],
        ["窗口帧率均值 (app FPS window mean)", mean_or_nan(app_fps_window_vals)],
        ["最新总帧率 (app FPS total latest)", app_fps_total_vals[-1] if app_fps_total_vals else math.nan],
        ["实时比例均值 (realtime ratio R mean)", mean_or_nan(realtime_vals)],
        ["实时比例最新值 (realtime ratio R latest)", realtime_vals[-1] if realtime_vals else math.nan],
        ["丢帧率均值 (drop rate mean, %)", mean_or_nan(drop_vals)],
        ["丢帧率最新值 (drop rate latest, %)", drop_vals[-1] if drop_vals else math.nan],
    ]
    lines.append(table(["指标", "数值"], summary_rows))
    lines.append("")

    lines.append("阶段延迟 (Stage Latency, ms)")
    lines.append("-" * 31)
    stage_rows = []
    for stage_name in metrics["stage_names"]:
        avg_vals = safe_values(records, "stage_ms", f"{stage_name}_avg")
        p95_vals = safe_values(records, "stage_ms", f"{stage_name}_p95")
        stage_rows.append(
            [
                stage_name,
                mean_or_nan(avg_vals),
                avg_vals[-1] if avg_vals else math.nan,
                mean_or_nan(p95_vals),
                p95_vals[-1] if p95_vals else math.nan,
            ]
        )
    lines.append(table(["阶段", "平均延迟均值", "平均延迟最新", "P95均值", "P95最新"], stage_rows))
    lines.append("")

    process_avg = safe_values(records, "stage_ms", "process_avg")
    process_p95 = safe_values(records, "stage_ms", "process_p95")
    loop_avg = safe_values(records, "stage_ms", "loop_avg")
    loop_p95 = safe_values(records, "stage_ms", "loop_p95")
    loop_max = safe_values(records, "stage_ms", "loop_max")
    latency_rows = [
        [
            "单次处理延迟 (process latency)",
            mean_or_nan(process_avg),
            process_avg[-1] if process_avg else math.nan,
            mean_or_nan(process_p95),
            process_p95[-1] if process_p95 else math.nan,
        ],
        [
            "单次迭代总延迟 (loop/iteration latency)",
            mean_or_nan(loop_avg),
            loop_avg[-1] if loop_avg else math.nan,
            mean_or_nan(loop_p95),
            loop_p95[-1] if loop_p95 else math.nan,
        ],
        [
            "单次迭代最大值 (loop max)",
            mean_or_nan(loop_max),
            loop_max[-1] if loop_max else math.nan,
            percentile(loop_max, 95.0),
            max(loop_max) if loop_max else math.nan,
        ],
    ]
    lines.append("端到端延迟 (End-to-End Latency, ms)")
    lines.append("-" * 38)
    lines.append(table(["指标", "平均均值", "平均最新", "P95/统计均值", "P95/统计最新"], latency_rows))
    lines.append("")

    detail_focus = (
        "palm_preprocess",
        "palm_preprocess_transform",
        "palm_input_load",
        "palm_inference",
        "palm_decode",
        "palm_accounted",
    )
    detail_rows = []
    for detail_name in detail_focus:
        avg_vals = safe_values(records, "palm_detail_ms", f"{detail_name}_avg")
        p95_vals = safe_values(records, "palm_detail_ms", f"{detail_name}_p95")
        if not avg_vals and not p95_vals:
            continue
        detail_rows.append(
            [
                detail_name,
                mean_or_nan(avg_vals),
                avg_vals[-1] if avg_vals else math.nan,
                mean_or_nan(p95_vals),
                p95_vals[-1] if p95_vals else math.nan,
                (avg_vals[-1] * k) if avg_vals else math.nan,
            ]
        )
    if detail_rows:
        lines.append("Palm 关键细分延迟 (Palm Detail Latency, ms)")
        lines.append("-" * 55)
        lines.append(
            table(
                ["阶段", "平均延迟均值", "平均延迟最新", "P95均值", "P95最新", "单次推理估计"],
                detail_rows,
            )
        )
        lines.append("")

    sensor_period = latest_summary.get("sensor_period_ms", as_float(config.get("sensor_period_ms", math.nan)))
    latency_latest = get_section(records[-1], "latency")
    realtime_rows = [
        ["传感器周期 (sensor period, ms)", sensor_period],
        ["循环P95周期数 (loop P95 / sensor period)", latency_latest.get("e2e_loop_p95_T", math.nan)],
        ["处理P95周期数 (process P95 / sensor period)", latency_latest.get("e2e_process_p95_T", math.nan)],
        ["延迟评分 (latency score)", latency_latest.get("latency_score_est_by_loop", math.nan)],
    ]
    lines.append("实时性 (Realtime)")
    lines.append("-" * 19)
    lines.append(table(["指标", "数值"], realtime_rows))
    lines.append("")

    lines.append("说明 (Notes)")
    lines.append("-" * 12)
    lines.append("avg 是设备端每个统计窗口内的每帧平均延迟，表中“均值”是对所有窗口再次求均值。")
    lines.append("P95 直接来自设备日志，不按 kInferInterval 换算。")
    lines.append("单次推理估计 (per-inference estimate) 仅对 avg 使用 avg * kInferInterval。")
    if mode == "palm_hand":
        lines.append("Hand 调用次数是按推理帧估计，日志未记录每帧实际 ROI 数量。")
    if parse_warnings or plot_warnings:
        lines.append("")
        lines.append("警告 (Warnings)")
        for warning in parse_warnings:
            lines.append(f"- {warning}")
        for warning in plot_warnings:
            lines.append(f"- {warning}")
    lines.append("")

    lines.append("输出文件 (Generated Files)")
    lines.append("-" * 26)
    lines.append(f"- {summary_txt_path}")
    lines.append(f"- {detail_txt_path}")
    for path in plot_paths:
        lines.append(f"- {path}")

    return "\n".join(lines)


def build_detailed_report(
    *,
    log_path: Path,
    out_dir: Path,
    mode: str,
    k: int,
    config: Dict[str, object],
    records: Sequence[Record],
    parse_warnings: Sequence[str],
    plot_paths: Sequence[Path],
    plot_warnings: Sequence[str],
    detail_txt_path: Path,
    summary_txt_path: Path,
) -> str:
    lines: List[str] = []
    lines.append("ssne_ai_demo Detailed Performance Statistics")
    lines.append("=" * 46)
    lines.append("")
    lines.append(f"Mode: {mode}")
    lines.append(f"kInferInterval: {k}")
    lines.append(f"Log file: {log_path}")
    lines.append(f"Output directory: {out_dir}")
    lines.append(f"Detailed TXT report: {detail_txt_path}")
    lines.append(f"Chinese summary report: {summary_txt_path}")
    lines.append("")

    if config:
        lines.append("Config")
        lines.append("-" * 6)
        config_rows = [[key, value] for key, value in sorted(config.items())]
        lines.append(table(["key", "value"], config_rows))
        lines.append("")

    if not records:
        lines.append("No [PERF][summary] records were found.")
        if parse_warnings:
            lines.append("")
            lines.append("Parse warnings")
            lines.extend(f"- {warning}" for warning in parse_warnings)
        return "\n".join(lines)

    metrics = collect_run_metrics(mode, k, config, records)
    app_fps_window_vals = safe_values(records, "summary", "app_fps_window")
    app_fps_total_vals = safe_values(records, "summary", "app_fps_total")
    realtime_vals = safe_values(records, "summary", "R")
    drop_vals = safe_values(records, "summary", "drop_rate_est_pct")
    loop_p95_vals = safe_values(records, "stage_ms", "loop_p95")
    process_p95_vals = safe_values(records, "stage_ms", "process_p95")

    lines.append("Summary")
    lines.append("-" * 7)
    summary_rows = [
        ["reported_windows", metrics["total_windows"]],
        ["reported_window_frames", metrics["total_reported_frames"]],
        ["latest_total_frames", metrics["last_total_frames"]],
        ["sensor_fps_cfg", metrics["sensor_fps"]],
        ["app_fps_window_mean", mean_or_nan(app_fps_window_vals)],
        ["app_fps_window_latest", app_fps_window_vals[-1] if app_fps_window_vals else math.nan],
        ["app_fps_total_latest", app_fps_total_vals[-1] if app_fps_total_vals else math.nan],
        ["realtime_ratio_R_mean", mean_or_nan(realtime_vals)],
        ["realtime_ratio_R_latest", realtime_vals[-1] if realtime_vals else math.nan],
        ["drop_rate_est_pct_mean", mean_or_nan(drop_vals)],
        ["drop_rate_est_pct_latest", drop_vals[-1] if drop_vals else math.nan],
        ["loop_p95_ms_mean", mean_or_nan(loop_p95_vals)],
        ["loop_p95_ms_latest", loop_p95_vals[-1] if loop_p95_vals else math.nan],
        ["process_p95_ms_mean", mean_or_nan(process_p95_vals)],
        ["process_p95_ms_latest", process_p95_vals[-1] if process_p95_vals else math.nan],
        ["palm_inference_count_est", metrics["total_palm_infer"]],
        ["hand_call_count_est", metrics["total_hand_calls"]],
    ]
    lines.append(table(["metric", "value"], summary_rows))
    lines.append("")

    lines.append("Important Notes")
    lines.append("-" * 15)
    lines.append(
        "- The C++ monitor reports per-window aggregates, so this script aggregates those window-level numbers."
    )
    lines.append(
        "- *_avg fields for skipped inference stages are per-frame averages; estimated per-inference avg is avg * kInferInterval."
    )
    lines.append(
        "- *_p95 fields are true window p95 values from the device log; this script does not multiply p95 by k."
    )
    if mode == "palm_hand":
        lines.append(
            "- hand_call_count_est counts frames where the cascade can run hand processing; exact hand model invocations per detected ROI are not logged."
        )
    else:
        lines.append("- palm mode treats hand_total timing as disabled.")
    lines.append("")

    lines.append("Window Summary")
    lines.append("-" * 14)
    window_rows = []
    for record in records:
        summary = get_section(record, "summary")
        window_rows.append(
            [
                record.get("index"),
                int(summary.get("frame", -1)),
                int(summary.get("window_frames", 0)),
                record.get("palm_infer_count", 0),
                record.get("hand_call_count_est", 0),
                summary.get("app_fps_window", math.nan),
                summary.get("R", math.nan),
                summary.get("drop_rate_est_pct", math.nan),
            ]
        )
    lines.append(
        table(
            ["win", "end_frame", "frames", "palm_inf", "hand_call_est", "fps_win", "R", "drop_%"],
            window_rows,
        )
    )
    lines.append("")

    lines.append("Stage Latency By Window (ms)")
    lines.append("-" * 28)
    stage_rows = []
    for record in records:
        stage = get_section(record, "stage_ms")
        row: List[object] = [record.get("index")]
        for stage_name in TOP_STAGE_ORDER:
            if mode == "palm" and stage_name == "hand_total":
                continue
            row.extend([stage.get(f"{stage_name}_avg", math.nan), stage.get(f"{stage_name}_p95", math.nan)])
        row.append(stage.get("loop_max", math.nan))
        stage_rows.append(row)

    stage_headers: List[str] = ["win"]
    for stage_name in TOP_STAGE_ORDER:
        if mode == "palm" and stage_name == "hand_total":
            continue
        stage_headers.extend([f"{stage_name}_avg", f"{stage_name}_p95"])
    stage_headers.append("loop_max")
    lines.append(table(stage_headers, stage_rows))
    lines.append("")

    lines.append("Palm Detail By Window (ms)")
    lines.append("-" * 27)
    detail_headers = ["win"]
    for detail_name in PALM_DETAIL_ORDER:
        detail_headers.extend([f"{detail_name}_avg", f"{detail_name}_p95"])
    detail_rows = []
    for record in records:
        detail = get_section(record, "palm_detail_ms")
        row = [record.get("index")]
        for detail_name in PALM_DETAIL_ORDER:
            row.extend([detail.get(f"{detail_name}_avg", math.nan), detail.get(f"{detail_name}_p95", math.nan)])
        detail_rows.append(row)
    lines.append(table(detail_headers, detail_rows))
    lines.append("")

    lines.append("Across-Window Aggregate: Summary Metrics")
    lines.append("-" * 40)
    lines.append(table(["metric", "mean", "p50", "p95", "min", "max", "latest"], aggregate_rows(records, "summary", SUMMARY_KEYS)))
    lines.append("")

    lines.append("Across-Window Aggregate: Stage Latency (ms)")
    lines.append("-" * 43)
    lines.append(table(["metric", "mean", "p50", "p95", "min", "max", "latest"], aggregate_rows(records, "stage_ms", stage_metric_keys(mode))))
    lines.append("")

    lines.append("Across-Window Aggregate: Palm Detail (ms)")
    lines.append("-" * 42)
    detail_keys: List[str] = []
    for name in PALM_DETAIL_ORDER:
        detail_keys.extend([f"{name}_avg", f"{name}_p95"])
    lines.append(table(["metric", "mean", "p50", "p95", "min", "max", "latest"], aggregate_rows(records, "palm_detail_ms", detail_keys)))
    lines.append("")

    lines.append("Estimated Per-Inference Average From Latest Window (ms)")
    lines.append("-" * 56)
    per_infer_rows = infer_avg_rows(records, mode, k)
    if per_infer_rows:
        lines.append(table(["stage", "per_frame_avg", "per_infer_avg_est", "method"], per_infer_rows))
    else:
        lines.append("No inference-stage avg fields were found.")
    lines.append("")

    if records[-1].get("latency"):
        lines.append("Across-Window Aggregate: Latency")
        lines.append("-" * 32)
        lines.append(table(["metric", "mean", "p50", "p95", "min", "max", "latest"], aggregate_rows(records, "latency", LATENCY_KEYS)))
        lines.append("")

    if records[-1].get("jitter"):
        lines.append("Across-Window Aggregate: Jitter")
        lines.append("-" * 31)
        lines.append(table(["metric", "mean", "p50", "p95", "min", "max", "latest"], aggregate_rows(records, "jitter", JITTER_KEYS)))
        lines.append("")

    lines.append("Generated Files")
    lines.append("-" * 15)
    lines.append(f"- {summary_txt_path}")
    lines.append(f"- {detail_txt_path}")
    for path in plot_paths:
        lines.append(f"- {path}")
    if not plot_paths:
        lines.append("- No SVG plots were generated.")
    lines.append("")

    if parse_warnings or plot_warnings:
        lines.append("Warnings")
        lines.append("-" * 8)
        for warning in parse_warnings:
            lines.append(f"- {warning}")
        for warning in plot_warnings:
            lines.append(f"- {warning}")
        lines.append("")

    return "\n".join(lines)


def configure_plot_style():
    try:
        import matplotlib.pyplot as plt
        from matplotlib.font_manager import FontProperties
    except Exception as exc:  # pragma: no cover - depends on local Python packages.
        return None, None, None, None, [f"matplotlib is unavailable: {exc}"]

    try:
        import seaborn as sns  # type: ignore

        sns.set_theme(style="whitegrid", context="notebook")
    except Exception:
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "ggplot")

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["axes.unicode_minus"] = False

    times_path = r"C:\Windows\Fonts\times.ttf"
    if os.path.exists(times_path):
        T_16 = FontProperties(fname=r"C:\Windows\Fonts\times.ttf", size=16)
        T_14 = FontProperties(fname=r"C:\Windows\Fonts\times.ttf", size=14)
        T_12 = FontProperties(fname=r"C:\Windows\Fonts\times.ttf", size=12)
    else:
        T_16 = FontProperties(family="Times New Roman", size=16)
        T_14 = FontProperties(family="Times New Roman", size=14)
        T_12 = FontProperties(family="Times New Roman", size=12)

    return plt, T_16, T_14, T_12, []


def set_tick_font(ax, font_prop) -> None:
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font_prop)


def save_figure(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, format="svg", bbox_inches="tight")


def positive_metric_mean(records: Sequence[Record], section: str, key: str) -> float:
    vals = [v for v in safe_values(records, section, key) if v > 0.0]
    return mean_or_nan(vals)


def zero_if_nan(value_ms: float) -> float:
    return 0.0 if math.isnan(value_ms) else value_ms


def add_positive_segment(segments: List[Tuple[str, float]], label: str, value_ms: float) -> None:
    if math.isnan(value_ms) or value_ms <= 0.0:
        return
    segments.append((label, value_ms))


def merge_tiny_segments(
    segments: Sequence[Tuple[str, float]],
    *,
    min_ms: float = 0.20,
    min_pct: float = 0.025,
) -> List[Tuple[str, float]]:
    total = sum(value_ms for _, value_ms in segments)
    if total <= 0.0:
        return []

    visible: List[Tuple[str, float]] = []
    tiny_total = 0.0
    for label, value_ms in segments:
        if value_ms < min_ms or value_ms / total < min_pct:
            tiny_total += value_ms
        else:
            visible.append((label, value_ms))

    if tiny_total > 0.0:
        visible.append(("Small stages", tiny_total))
    return visible


def build_iteration_p95_breakdown_segments(
    records: Sequence[Record],
    mode: str,
) -> Tuple[List[Tuple[str, float]], float]:
    palm_total_p95 = positive_metric_mean(records, "stage_ms", "palm_total_p95")
    hand_total_p95 = positive_metric_mean(records, "stage_ms", "hand_total_p95") if mode == "palm_hand" else math.nan
    draw_p95 = positive_metric_mean(records, "stage_ms", "draw_p95")
    get_image_p95 = positive_metric_mean(records, "stage_ms", "get_image_p95")
    process_p95 = positive_metric_mean(records, "stage_ms", "process_p95")
    loop_p95 = positive_metric_mean(records, "stage_ms", "loop_p95")

    palm_transform = positive_metric_mean(records, "palm_detail_ms", "palm_preprocess_transform_p95")
    palm_tensor_load = (
        zero_if_nan(positive_metric_mean(records, "palm_detail_ms", "palm_preprocess_manual_load_p95"))
        + zero_if_nan(positive_metric_mean(records, "palm_detail_ms", "palm_input_load_p95"))
    )
    palm_inference = positive_metric_mean(records, "palm_detail_ms", "palm_inference_p95")
    palm_post = (
        zero_if_nan(positive_metric_mean(records, "palm_detail_ms", "palm_getoutput_p95"))
        + zero_if_nan(positive_metric_mean(records, "palm_detail_ms", "palm_output_meta_p95"))
        + zero_if_nan(positive_metric_mean(records, "palm_detail_ms", "palm_decode_p95"))
        + zero_if_nan(positive_metric_mean(records, "palm_detail_ms", "palm_verbose_log_p95"))
    )
    palm_known = zero_if_nan(palm_transform) + palm_tensor_load + zero_if_nan(palm_inference) + palm_post
    palm_other = max(palm_total_p95 - palm_known, 0.0) if not math.isnan(palm_total_p95) else math.nan

    process_known = sum(
        value_ms
        for value_ms in (palm_total_p95, hand_total_p95 if mode == "palm_hand" else 0.0, draw_p95)
        if not math.isnan(value_ms)
    )
    process_other = max(process_p95 - process_known, 0.0) if not math.isnan(process_p95) else math.nan
    loop_other = (
        max(loop_p95 - get_image_p95 - process_p95, 0.0)
        if not any(math.isnan(v) for v in (loop_p95, get_image_p95, process_p95))
        else math.nan
    )

    segments: List[Tuple[str, float]] = []
    add_positive_segment(segments, "Get image", get_image_p95)
    add_positive_segment(segments, "Palm transform", palm_transform)
    add_positive_segment(segments, "Palm tensor load", palm_tensor_load)
    add_positive_segment(segments, "Palm inference", palm_inference)
    add_positive_segment(segments, "Palm postprocess", palm_post)
    add_positive_segment(segments, "Palm other", palm_other)
    if mode == "palm_hand":
        add_positive_segment(segments, "Hand total", hand_total_p95)
    add_positive_segment(segments, "Draw", draw_p95)
    add_positive_segment(segments, "Process overhead", process_other)
    add_positive_segment(segments, "Loop overhead", loop_other)

    total_ms = sum(value_ms for _, value_ms in segments)
    return merge_tiny_segments(segments), total_ms


def build_iteration_pie_info_text(
    *,
    mode: str,
    segments: Sequence[Tuple[str, float]],
    total_ms: float,
    colors: Sequence[str],
) -> str:
    lines: List[str] = []
    lines.append("P95 Inference Iteration Breakdown")
    lines.append("=" * 35)
    lines.append("")
    lines.append(f"Mode: {mode}")
    lines.append("Figure: pie chart with in-slice labels and percentages, without title or right-side legend.")
    lines.append("Data source: all [PERF] windows, positive-window mean of each P95 metric.")
    lines.append("Unit: ms")
    lines.append(f"Sum of shown P95 components: {total_ms:.3f} ms")
    lines.append("")
    lines.append(
        "Note: the sum is useful for bottleneck composition in slides, but it is not the strict "
        "end-to-end P95 of one exact frame because each stage P95 may come from a different frame."
    )
    lines.append("")
    lines.append(table(["index", "label", "color", "latency_ms", "percent"], []))
    rows = []
    for idx, (label, value_ms) in enumerate(segments, start=1):
        percent = 100.0 * value_ms / total_ms if total_ms > 0.0 else math.nan
        color = colors[(idx - 1) % len(colors)]
        rows.append([idx, label, color, value_ms, f"{percent:.1f}%"])
    if rows:
        lines[-1] = table(["index", "label", "color", "latency_ms", "percent"], rows)
    return "\n".join(lines)


def make_pie_label_formatter(labels: Sequence[str]):
    state = {"index": 0}

    def formatter(percent: float) -> str:
        index = state["index"]
        state["index"] += 1
        label = labels[index] if index < len(labels) else ""
        return f"{label}\n{percent:.1f}%"

    return formatter


def plot_stats(records: Sequence[Record], mode: str, k: int, out_prefix: Path) -> Tuple[List[Path], List[str]]:
    plt, T_16, T_14, T_12, warnings = configure_plot_style()
    if plt is None:
        return [], warnings

    plot_paths: List[Path] = []
    windows = [int(record.get("index", i + 1)) for i, record in enumerate(records)]
    if not records:
        return plot_paths, warnings

    try:
        fps = safe_values(records, "summary", "app_fps_window")
        realtime = safe_values(records, "summary", "R")
        drop = safe_values(records, "summary", "drop_rate_est_pct")
        if fps:
            fig, ax1 = plt.subplots(figsize=(8.6, 4.8))
            ax1.plot(windows[: len(fps)], fps, marker="o", linewidth=2.0, color="#276FBF", label="App FPS")
            ax1.set_xlabel("Window", fontproperties=T_14)
            ax1.set_ylabel("FPS", fontproperties=T_14)
            ax1.set_title("Runtime Throughput", fontproperties=T_16)
            ax1.grid(True, alpha=0.25)

            ax2 = ax1.twinx()
            if realtime:
                ax2.plot(windows[: len(realtime)], realtime, marker="s", linewidth=1.8, color="#2A9D8F", label="Realtime ratio")
            if drop:
                ax2.plot(windows[: len(drop)], [v / 100.0 for v in drop], marker="^", linewidth=1.8, color="#E76F51", label="Drop rate")
            ax2.set_ylabel("Ratio", fontproperties=T_14)
            set_tick_font(ax1, T_12)
            set_tick_font(ax2, T_12)
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", prop=T_12)
            path = out_prefix.with_name(out_prefix.name + "_throughput.svg")
            save_figure(fig, path)
            plt.close(fig)
            plot_paths.append(path)

        latest = records[-1]
        stage = get_section(latest, "stage_ms")
        stage_names = [name for name in TOP_STAGE_ORDER if not (mode == "palm" and name == "hand_total")]
        p95_values = [stage.get(f"{name}_p95", math.nan) for name in stage_names]
        avg_values = [stage.get(f"{name}_avg", math.nan) for name in stage_names]
        if any(not math.isnan(v) for v in p95_values):
            x = list(range(len(stage_names)))
            width = 0.38
            fig, ax = plt.subplots(figsize=(8.6, 4.8))
            ax.bar([i - width / 2 for i in x], avg_values, width=width, color="#5B8DEF", label="Average")
            ax.bar([i + width / 2 for i in x], p95_values, width=width, color="#F08A5D", label="P95")
            ax.set_xticks(x)
            ax.set_xticklabels(stage_names, fontproperties=T_12, rotation=20, ha="right")
            ax.set_ylabel("Latency (ms)", fontproperties=T_14)
            ax.set_title("Latest Window Stage Latency", fontproperties=T_16)
            ax.legend(prop=T_12)
            ax.grid(axis="y", alpha=0.25)
            set_tick_font(ax, T_12)
            path = out_prefix.with_name(out_prefix.name + "_stage_latency.svg")
            save_figure(fig, path)
            plt.close(fig)
            plot_paths.append(path)

        detail = get_section(latest, "palm_detail_ms")
        detail_names = [name for name in PALM_DETAIL_ORDER if f"{name}_p95" in detail]
        detail_p95 = [detail.get(f"{name}_p95", math.nan) for name in detail_names]
        detail_avg = [detail.get(f"{name}_avg", math.nan) for name in detail_names]
        if detail_names:
            y = list(range(len(detail_names)))
            fig, ax = plt.subplots(figsize=(8.8, 5.6))
            ax.barh([i - 0.18 for i in y], detail_avg, height=0.35, color="#6A994E", label="Average")
            ax.barh([i + 0.18 for i in y], detail_p95, height=0.35, color="#BC4749", label="P95")
            ax.set_yticks(y)
            ax.set_yticklabels(detail_names, fontproperties=T_12)
            ax.invert_yaxis()
            ax.set_xlabel("Latency (ms)", fontproperties=T_14)
            ax.set_title("Latest Window Palm Detail", fontproperties=T_16)
            ax.legend(prop=T_12)
            ax.grid(axis="x", alpha=0.25)
            set_tick_font(ax, T_12)
            path = out_prefix.with_name(out_prefix.name + "_palm_detail.svg")
            save_figure(fig, path)
            plt.close(fig)
            plot_paths.append(path)

        pie_segments, pie_total_ms = build_iteration_p95_breakdown_segments(records, mode)
        if pie_segments:
            labels = [label for label, _ in pie_segments]
            sizes = [value_ms for _, value_ms in pie_segments]
            colors = [
                "#4E79A7",
                "#F28E2B",
                "#59A14F",
                "#E15759",
                "#76B7B2",
                "#EDC948",
                "#B07AA1",
                "#FF9DA7",
                "#9C755F",
                "#BAB0AC",
            ]
            fig, ax = plt.subplots(figsize=(4.9, 4.9))
            _, _, autotexts = ax.pie(
                sizes,
                colors=colors[: len(sizes)],
                startangle=90,
                counterclock=False,
                autopct=make_pie_label_formatter(labels),
                pctdistance=0.67,
                radius=1.24,
                textprops={"color": "black", "ha": "center", "va": "center"},
                wedgeprops={"linewidth": 1.0, "edgecolor": "white"},
            )
            ax.axis("equal")
            ax.set_axis_off()
            for text in autotexts:
                text.set_fontproperties(T_14)
            path = out_prefix.with_name(out_prefix.name + "_iteration_pie.svg")
            info_path = out_prefix.with_name(out_prefix.name + "_iteration_pie_info.txt")
            info_path.write_text(
                build_iteration_pie_info_text(
                    mode=mode,
                    segments=pie_segments,
                    total_ms=pie_total_ms,
                    colors=colors,
                )
                + "\n",
                encoding="utf-8",
            )
            fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
            save_figure(fig, path)
            plt.close(fig)
            plot_paths.append(path)
            plot_paths.append(info_path)

        timeline_stages = ["palm_total", "process", "loop"]
        if mode == "palm_hand":
            timeline_stages.insert(1, "hand_total")
        fig, ax = plt.subplots(figsize=(8.8, 4.8))
        plotted = False
        colors = {
            "palm_total": "#276FBF",
            "hand_total": "#8E5CF7",
            "process": "#F4A261",
            "loop": "#D62828",
        }
        for name in timeline_stages:
            vals = safe_values(records, "stage_ms", f"{name}_p95")
            if vals:
                ax.plot(windows[: len(vals)], vals, marker="o", linewidth=1.8, color=colors.get(name), label=f"{name} p95")
                plotted = True
        if plotted:
            ax.set_xlabel("Window", fontproperties=T_14)
            ax.set_ylabel("Latency P95 (ms)", fontproperties=T_14)
            ax.set_title("P95 Latency Timeline", fontproperties=T_16)
            ax.legend(prop=T_12)
            ax.grid(True, alpha=0.25)
            set_tick_font(ax, T_12)
            path = out_prefix.with_name(out_prefix.name + "_p95_timeline.svg")
            save_figure(fig, path)
            plt.close(fig)
            plot_paths.append(path)
        else:
            plt.close(fig)

    except Exception as exc:  # pragma: no cover - defensive for odd local plot setups.
        warnings.append(f"plot generation failed: {exc}")

    return plot_paths, warnings


def main() -> int:
    args = parse_args()
    log_path = args.log.expanduser().resolve()
    if not log_path.exists():
        print(f"Log file not found: {log_path}", file=sys.stderr)
        return 2
    if not log_path.is_file():
        print(f"Log path is not a file: {log_path}", file=sys.stderr)
        return 2

    out_dir = (args.out_dir.expanduser().resolve() if args.out_dir else log_path.parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = out_dir / f"{log_path.stem}_stats"
    summary_txt_path = out_prefix.with_name(out_prefix.name + "_summary.txt")
    detail_txt_path = out_prefix.with_name(out_prefix.name + "_detail.txt")

    config, records, parse_warnings = parse_perf_log(log_path)
    add_inference_counts(records, args.mode, args.kInferInterval)
    plot_paths, plot_warnings = plot_stats(records, args.mode, args.kInferInterval, out_prefix)

    summary_report = build_chinese_summary_report(
        log_path=log_path,
        out_dir=out_dir,
        mode=args.mode,
        k=args.kInferInterval,
        config=config,
        records=records,
        parse_warnings=parse_warnings,
        plot_paths=plot_paths,
        plot_warnings=plot_warnings,
        detail_txt_path=detail_txt_path,
        summary_txt_path=summary_txt_path,
    )
    detailed_report = build_detailed_report(
        log_path=log_path,
        out_dir=out_dir,
        mode=args.mode,
        k=args.kInferInterval,
        config=config,
        records=records,
        parse_warnings=parse_warnings,
        plot_paths=plot_paths,
        plot_warnings=plot_warnings,
        detail_txt_path=detail_txt_path,
        summary_txt_path=summary_txt_path,
    )
    print(summary_report)
    print("")
    print(detailed_report)
    summary_txt_path.write_text(summary_report + "\n", encoding="utf-8")
    detail_txt_path.write_text(detailed_report + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
