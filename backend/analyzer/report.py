from __future__ import annotations

from html import escape


def generate_batch_report(models: dict[str, object], output_format: str = "json") -> dict:
    from .compare import compare_models

    rows: list[dict] = []
    for model_id, model in models.items():
        op_types = sorted({layer.op_type for layer in model.layers})
        rows.append({
            "model_id": model_id,
            "format": model.format,
            "producer": model.producer,
            "layer_count": len(model.layers),
            "param_count": model.total_params(),
            "op_types": op_types,
        })

    comparisons: list[dict] = []
    ids = list(models.keys())
    for index in range(len(ids) - 1):
        a_id, b_id = ids[index], ids[index + 1]
        comparisons.append(compare_models(models[a_id], models[b_id], a_id, b_id)["summary"])

    report: dict[str, object] = {
        "format": output_format,
        "models": rows,
        "comparisons": comparisons,
        "summary": {
            "model_count": len(rows),
            "total_params": sum(row["param_count"] for row in rows),
            "max_layers": max((row["layer_count"] for row in rows), default=0),
        },
    }
    if output_format == "html":
        report["html"] = _render_html(rows, report["summary"])
    return report


def _render_html(rows: list[dict], summary: dict) -> str:
    body_rows = "\n".join(
        "<tr>"
        f"<td>{escape(str(row['model_id']))}</td>"
        f"<td>{escape(str(row['format']))}</td>"
        f"<td>{row['layer_count']}</td>"
        f"<td>{row['param_count']}</td>"
        f"<td>{escape(', '.join(row['op_types']))}</td>"
        "</tr>"
        for row in rows
    )
    return f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>ModelViz Report</title></head>
<body>
  <h1>ModelViz Batch Report</h1>
  <p>Models: {summary['model_count']} | Total params: {summary['total_params']}</p>
  <table border="1" cellspacing="0" cellpadding="6">
    <thead><tr><th>ID</th><th>Format</th><th>Layers</th><th>Params</th><th>Ops</th></tr></thead>
    <tbody>{body_rows}</tbody>
  </table>
</body>
</html>"""
