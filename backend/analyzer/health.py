from __future__ import annotations

import numpy as np


def health_check(ir_model, model_id: str = "") -> dict:
    issues: list[dict] = []

    for layer in ir_model.layers:
        _check_weight_outliers(layer, issues)
        _check_high_sparsity(layer, issues)
        if "batchnorm" in layer.op_type.lower():
            _check_bn_anomaly(layer, issues)
        if layer.weights:
            _check_vanishing_gradient_risk(layer, issues)

    critical = len([i for i in issues if i["severity"] == "critical"])
    warning = len([i for i in issues if i["severity"] == "warning"])

    return {
        "model_id": model_id,
        "issues": issues,
        "summary": {"total_issues": len(issues), "critical_count": critical, "warning_count": warning},
    }


def _check_weight_outliers(layer, issues: list[dict]) -> None:
    for wname, warr in layer.weights.items():
        if not hasattr(warr, "shape") or warr.size < 10:
            continue
        arr = np.asarray(warr, dtype=np.float32)
        mean = float(arr.mean())
        std = float(arr.std())
        if std < 1e-8:
            continue
        zscores = np.abs((arr - mean) / std)
        outlier_mask = zscores > 5.0
        n_outliers = int(outlier_mask.sum())
        if n_outliers > 0:
            max_sigma = float(zscores.max())
            severity = "critical" if n_outliers > max(10, arr.size * 0.01) else "warning"
            issues.append({
                "layer": layer.name,
                "severity": severity,
                "type": "weight_outlier",
                "message": f"{wname}: 检测到 {n_outliers} 个权重离群值 (最大 {max_sigma:.1f}σ)",
                "detail": {"weight_name": wname, "outlier_count": n_outliers, "max_sigma": round(max_sigma, 2)},
            })


def _check_high_sparsity(layer, issues: list[dict]) -> None:
    for wname, warr in layer.weights.items():
        if not hasattr(warr, "shape") or warr.size < 10:
            continue
        arr = np.asarray(warr, dtype=np.float32)
        sparsity = float(np.sum(np.abs(arr) < 1e-7) / arr.size)
        if sparsity > 0.5:
            severity = "critical" if sparsity > 0.9 else "warning"
            issues.append({
                "layer": layer.name,
                "severity": severity,
                "type": "high_sparsity",
                "message": f"{wname}: 稀疏度 {sparsity:.1%} (近零权重占比)",
                "detail": {"weight_name": wname, "sparsity": round(sparsity, 4)},
            })


def _check_bn_anomaly(layer, issues: list[dict]) -> None:
    running_mean = layer.weights.get("running_mean")
    running_var = layer.weights.get("running_var")
    if running_mean is not None and hasattr(running_mean, "shape"):
        mean_abs = float(np.mean(np.abs(np.asarray(running_mean, dtype=np.float32))))
        if mean_abs > 3.0:
            issues.append({
                "layer": layer.name,
                "severity": "warning",
                "type": "bn_anomaly",
                "message": f"BatchNorm running_mean 偏离零点 (mean|abs|={mean_abs:.2f})，归一化效果可能减弱",
                "detail": {"mean_abs": round(mean_abs, 2), "param": "running_mean"},
            })
    if running_var is not None and hasattr(running_var, "shape"):
        var_mean = float(np.mean(np.asarray(running_var, dtype=np.float32)))
        if var_mean < 1e-3:
            issues.append({
                "layer": layer.name,
                "severity": "critical",
                "type": "bn_anomaly",
                "message": f"BatchNorm running_var 极小 ({var_mean:.6f})，可能导致数值不稳定",
                "detail": {"var_mean": round(var_mean, 6), "param": "running_var"},
            })
        elif var_mean > 5.0:
            issues.append({
                "layer": layer.name,
                "severity": "warning",
                "type": "bn_anomaly",
                "message": f"BatchNorm running_var 偏大 ({var_mean:.2f})，训练可能未充分收敛",
                "detail": {"var_mean": round(var_mean, 2), "param": "running_var"},
            })


def _check_vanishing_gradient_risk(layer, issues: list[dict]) -> None:
    for wname, warr in layer.weights.items():
        if not hasattr(warr, "shape") or warr.size < 10:
            continue
        arr = np.asarray(warr, dtype=np.float32).ravel()
        std = float(np.std(arr))
        rng = float(np.max(arr) - np.min(arr))
        if rng < 1e-8:
            continue
        if std / rng < 0.05 and std < 1e-3:
            issues.append({
                "layer": layer.name,
                "severity": "warning",
                "type": "vanishing_gradient_risk",
                "message": f"{wname}: 权重分布极窄 (σ={std:.6f}, σ/range={std/rng:.4f})，可能存在梯度消失风险",
                "detail": {"weight_name": wname, "std": round(std, 6), "std_range_ratio": round(std / rng, 4)},
            })
