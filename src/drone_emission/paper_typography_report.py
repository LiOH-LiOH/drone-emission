"""Readable and structured typography audit from formal export records."""

import json
from pathlib import Path

from .visualization.typography import load_typography, typography_signature


def write_typography_summary(figures_root):
    root = Path(figures_root)
    report_root = root / "paper" / "report"
    records, exceptions, stale = [], [], []
    for path in sorted((report_root / "typography").glob("*.json")):
        document = json.loads(path.read_text())
        if document["typography_sha256"] != typography_signature():
            stale.append(document["output"])
            continue
        sizes = [text["size_pt"] for text in document["texts"]]
        records.append({"output": document["output"], "text_count": len(sizes),
                        "min_size_pt": min(sizes, default=None), "max_size_pt": max(sizes, default=None)})
        for text in document["texts"]:
            if text["size_pt"] < 7 or text.get("exception_reason"):
                exceptions.append({"output": document["output"], **text})
    masters = {path.stem: json.loads(path.read_text()) for path in sorted((root / "inkscape" / "typography").glob("*.json"))}
    document = {"typography_sha256": typography_signature(), "config": load_typography(),
                "subfigures": records, "exceptions": exceptions, "stale_reports": stale, "masters": masters}
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "typography_summary.json").write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n")
    lines = ["# 字体检查报告", "", "正式 PDF 使用 Arial；子图以原尺寸进入拼版。", "",
             "基准：轴标题 9 pt，刻度/图例/主要标注 8 pt，次要标注 7 pt，面板编号 11 pt。", "",
             "| 子图 | 文字数 | 最小字号 (pt) | 最大字号 (pt) |", "|---|---:|---:|---:|"]
    for item in records:
        lines.append(f"| {item['output']} | {item['text_count']} | {item['min_size_pt']} | {item['max_size_pt']} |")
    lines += ["", "## 小字号与其他例外", ""]
    if not exceptions:
        lines.append("无已记录例外。")
    else:
        lines += ["| 子图 | 用途 | 文字 | 字号 (pt) | 原因 |", "|---|---|---|---:|---|"]
        for item in exceptions:
            label = item["text"].replace("\n", " / ").replace("|", "\\|")
            lines.append(f"| {item['output']} | {item['role']} | {label} | {item['size_pt']} | {item['exception_reason'] or '未说明，需修复'} |")
    if stale:
        lines += ["", "以下报告已过期，需要重绘：" + ", ".join(stale)]
    lines += ["", "此报告记录字体设置；重叠、裁切和最终文字效果还需结合单图预览检查。", ""]
    (report_root / "typography_summary.md").write_text("\n".join(lines))
