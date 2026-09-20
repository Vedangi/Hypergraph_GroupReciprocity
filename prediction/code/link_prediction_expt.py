""" Temporal link-prediction experiment for directed hypergraphs.

Targets
-------
``y_dyad``
    Does recipient ``v`` send to the original sender ``s`` in any form within
    the horizon?
``y_mode``
    Among rows where ``v`` responds to ``s``, is there a strict group response
    that re-engages an original co-recipient and reaches overlap ``theta``?
``y_group_exact``
    Does ``v`` lead a response with exactly the original node set?

Primary ablation
----------------
For every label and horizon, compare:

* GRAPH
* GRAPH + SIZE
* GRAPH + GROUP_NO_SIZE
* FULL_HYPERGRAPH

Feature rows are built once and relabeled for every horizon.  Feature
selection uses train -> validation only; the final test period is not used to
choose features.  Test uncertainty is estimated by paired cluster bootstrap
over focal events.

Example
-------
python link_prediction_expt.py --dataset emaileu --horizons 1,2,7,30
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from link_pred_features import (
    ALL_FEATURES,
    DATASETS,
    GRAPH_FEATURES,
    HYPERGRAPH_FEATURES,
    build_causal_rows,
    load_dataset,
)


LABELS = ("y_dyad", "y_mode", "y_group_exact")

METRIC_COLUMNS = [
    "dataset", "horizon", "unit", "label", "model", "feature_set",
    "n_features", "features", "n_train", "n_validation", "n_test",
    "test_positives", "test_prevalence", "roc_auc", "average_precision",
    "relative_ap", "delta_ap_vs_graph", "delta_ap_ci_low",
    "delta_ap_ci_high",
]
LOO_COLUMNS = [
    "dataset", "horizon", "unit", "label", "feature",
    "validation_ap_full", "validation_ap_without", "delta_when_removed",
    "selected", "selection_reason", "validation_ap_reduced_final",
]
COEFFICIENT_COLUMNS = [
    "dataset", "horizon", "unit", "label", "feature", "coefficient",
    "abs_coefficient",
]


# ---------------------------------------------------------------------------
# Horizon labels and temporal partitions

def add_labels(delay_frame, horizon, t_end):
    """Derive the three labels and remove right-censored focal rows."""
    frame = delay_frame[
        delay_frame["t"].to_numpy() <= t_end - horizon
    ].copy()
    dyad = (frame["d_dyad"].to_numpy() <= horizon).astype(np.int8)
    overlap = (frame["d_overlap"].to_numpy() <= horizon).astype(np.int8)
    exact = (frame["d_exact"].to_numpy() <= horizon).astype(np.int8)

    frame["y_dyad"] = dyad
    frame["y_group_overlap"] = overlap
    frame["y_group_exact"] = exact
    # A strict overlap response necessarily includes s, hence is also dyadic.
    frame["y_mode"] = np.where(dyad == 1, overlap, -1).astype(np.int8)
    return frame.reset_index(drop=True)


def temporal_cutoffs(times, train_q=0.50, val_q=0.75):
    """Choose fixed train/validation cut times from the causal sample stream."""
    if not 0 < train_q < val_q < 1:
        raise ValueError("Require 0 < train_q < val_q < 1")
    train_cut, val_cut = np.quantile(times, [train_q, val_q])
    return float(train_cut), float(val_cut)


def temporal_masks(frame, horizon, train_cut, val_cut):
    """Return embargoed train, validation, and test masks."""
    times = frame["t"].to_numpy()
    train = (times + horizon) <= train_cut
    valid = (times >= train_cut) & ((times + horizon) <= val_cut)
    test = times >= val_cut
    return train, valid, test


def label_population(frame, label, train, valid, test):
    """Apply the population restriction associated with one label."""
    if label == "y_mode":
        eligible = frame[label].to_numpy() >= 0
    elif label == "y_group_exact":
        eligible = frame["k"].to_numpy() >= 3
    else:
        eligible = np.ones(len(frame), dtype=bool)
    return train & eligible, valid & eligible, test & eligible


# ---------------------------------------------------------------------------
# Models and scores

def make_logreg():
    return (
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=1.0,
        ),
        True,
    )


def make_lgbm(pruning=False):
    import lightgbm as lgb

    return (
        lgb.LGBMClassifier(
            n_estimators=100 if pruning else 300,
            learning_rate=0.06 if pruning else 0.04,
            num_leaves=31,
            min_child_samples=30,
            subsample=0.85,
            colsample_bytree=0.9,
            class_weight="balanced",
            random_state=7,
            n_jobs=-1,
            verbosity=-1,
        ),
        False,
    )


def available_models():
    models = [("LogReg", make_logreg)]
    try:
        import lightgbm  # noqa: F401
        models.append(("LightGBM", make_lgbm))
    except ImportError:
        print("  LightGBM is unavailable; running logistic regression only.")
    return models


def fit_predict(frame, label, train, test, features, model_fn):
    """Fit on ``train`` and return test labels/probabilities, or ``None``."""
    if train.sum() == 0 or test.sum() == 0 or not features:
        return None
    y_train = frame[label].to_numpy()[train]
    y_test = frame[label].to_numpy()[test]
    if (
        y_train.sum() in (0, len(y_train))
        or y_test.sum() in (0, len(y_test))
    ):
        return None

    model, scale = model_fn()
    x_train = frame.loc[train, features].to_numpy()
    x_test = frame.loc[test, features].to_numpy()
    if scale:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)[:, 1]
    return y_test, probabilities


def prediction_scores(result):
    if result is None:
        return float("nan"), float("nan")
    labels, probabilities = result
    return (
        roc_auc_score(labels, probabilities),
        average_precision_score(labels, probabilities),
    )


def paired_event_bootstrap(
    y,
    p_base,
    p_alt,
    event_ids,
    n_boot=200,
    seed=0,
):
    """Paired 95% CI for AP(alt)-AP(base), clustered by focal event."""
    if not n_boot:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    unique, inverse = np.unique(event_ids, return_inverse=True)
    order = np.argsort(inverse, kind="mergesort")
    bounds = np.searchsorted(
        inverse[order], np.arange(len(unique) + 1))
    groups = [
        order[bounds[i]:bounds[i + 1]]
        for i in range(len(unique))
    ]

    deltas = []
    for _ in range(n_boot):
        chosen = rng.integers(0, len(groups), len(groups))
        indices = np.concatenate([groups[i] for i in chosen])
        y_boot = y[indices]
        if y_boot.sum() in (0, len(y_boot)):
            continue
        deltas.append(
            average_precision_score(y_boot, p_alt[indices])
            - average_precision_score(y_boot, p_base[indices])
        )
    if len(deltas) < max(20, n_boot // 4):
        return float("nan"), float("nan")
    return (
        float(np.percentile(deltas, 2.5)),
        float(np.percentile(deltas, 97.5)),
    )


# ---------------------------------------------------------------------------
# Validation-only feature selection and importance

def _subsample_mask(mask, maximum, seed):
    indices = np.flatnonzero(mask)
    if maximum and len(indices) > maximum:
        rng = np.random.default_rng(seed)
        indices = np.sort(
            rng.choice(indices, size=maximum, replace=False))
    output = np.zeros(len(mask), dtype=bool)
    output[indices] = True
    return output


def _validation_ap(frame, label, train, valid, features, model_fn):
    return prediction_scores(
        fit_predict(frame, label, train, valid, features, model_fn)
    )[1]


def loo_ablation(
    frame,
    label,
    train,
    valid,
    candidates,
    tolerance=0.001,
    max_train=250_000,
    max_valid=100_000,
    model_fn=None,
):
    """Validation-only leave-one-out screening with joint recovery.

    ``delta_when_removed = AP(full)-AP(without feature)``:

    * positive: removal hurts, so the feature contributes unique signal;
    * near zero: redundant with the remaining features;
    * negative: validation AP improves when the feature is removed.

    Individually redundant correlated features can all be dropped by naive
    LOO.  The recovery pass adds omitted features back until the reduced set is
    within ``tolerance`` of complete-set validation AP.
    """
    if model_fn is None:
        try:
            import lightgbm  # noqa: F401
            model_fn = lambda: make_lgbm(pruning=True)
        except ImportError:
            model_fn = make_logreg

    selection_train = _subsample_mask(train, max_train, seed=11)
    selection_valid = _subsample_mask(valid, max_valid, seed=17)
    complete = list(dict.fromkeys(candidates))
    full_ap = _validation_ap(
        frame,
        label,
        selection_train,
        selection_valid,
        complete,
        model_fn,
    )

    rows = []
    for feature in complete:
        without = [name for name in complete if name != feature]
        ap_without = _validation_ap(
            frame,
            label,
            selection_train,
            selection_valid,
            without,
            model_fn,
        )
        delta = (
            full_ap - ap_without
            if np.isfinite(full_ap) and np.isfinite(ap_without)
            else float("nan")
        )
        selected = bool(np.isfinite(delta) and delta > tolerance)
        rows.append({
            "feature": feature,
            "validation_ap_full": full_ap,
            "validation_ap_without": ap_without,
            "delta_when_removed": delta,
            "selected": selected,
            "selection_reason": (
                "individual_loo" if selected else "omitted"
            ),
        })

    selected = [
        row["feature"] for row in rows if row["selected"]
    ]
    omitted = sorted(
        [row for row in rows if not row["selected"]],
        key=lambda row: (
            row["delta_when_removed"]
            if np.isfinite(row["delta_when_removed"])
            else -np.inf
        ),
        reverse=True,
    )

    if not np.isfinite(full_ap):
        # Selection is not possible when validation has only one class.
        for row in rows:
            row["selected"] = True
            row["selection_reason"] = "selection_unavailable_keep_all"
            row["validation_ap_reduced_final"] = float("nan")
        return complete, rows

    if not selected and omitted:
        first = omitted.pop(0)
        first["selected"] = True
        first["selection_reason"] = "minimum_nonempty"
        selected.append(first["feature"])

    reduced_ap = _validation_ap(
        frame,
        label,
        selection_train,
        selection_valid,
        selected,
        model_fn,
    )
    while (
        omitted
        and (
            not np.isfinite(reduced_ap)
            or reduced_ap < full_ap - tolerance
        )
    ):
        add_back = omitted.pop(0)
        add_back["selected"] = True
        add_back["selection_reason"] = "joint_recovery"
        selected.append(add_back["feature"])
        reduced_ap = _validation_ap(
            frame,
            label,
            selection_train,
            selection_valid,
            selected,
            model_fn,
        )

    for row in rows:
        row["validation_ap_reduced_final"] = reduced_ap
    return selected, rows


def standardized_logreg_coefficients(frame, label, mask, features):
    """Return standardized logistic coefficients fitted without test rows."""
    y = frame[label].to_numpy()[mask]
    if mask.sum() == 0 or y.sum() in (0, len(y)):
        return []
    scaler = StandardScaler()
    x = scaler.fit_transform(frame.loc[mask, features].to_numpy())
    model = LogisticRegression(
        max_iter=2000, class_weight="balanced")
    model.fit(x, y)
    return [
        {
            "feature": feature,
            "coefficient": float(coefficient),
            "abs_coefficient": float(abs(coefficient)),
        }
        for feature, coefficient in zip(features, model.coef_[0])
    ]


# ---------------------------------------------------------------------------
# Unified horizon and feature-set ablation

def feature_sets(selected=None, extra_candidates=()):
    size = ["h_rsize"]
    group_without_size = [
        feature for feature in HYPERGRAPH_FEATURES
        if feature != "h_rsize"
    ]
    sets = {
        "GRAPH": GRAPH_FEATURES,
        "GRAPH+SIZE": GRAPH_FEATURES + size,
        "GRAPH+GROUP_NO_SIZE": (
            GRAPH_FEATURES + group_without_size
        ),
        "FULL_HYPERGRAPH": (
            GRAPH_FEATURES + HYPERGRAPH_FEATURES
        ),
    }
    if extra_candidates:
        sets["FULL+CANDIDATES"] = (
            GRAPH_FEATURES + HYPERGRAPH_FEATURES + list(extra_candidates)
        )
    if selected:
        sets["SELECTED"] = selected
    return sets


def evaluate_horizon(
    dataset,
    delay_frame,
    horizon_value,
    horizon_native,
    unit,
    t_end,
    train_cut,
    val_cut,
    models,
    do_selection=True,
    prune_tol=0.001,
    selection_max_train=250_000,
    selection_max_valid=100_000,
    n_boot=200,
    extra_candidates=(),
):
    """Evaluate all three labels and return metric/LOO/coefficient rows."""
    frame = add_labels(delay_frame, horizon_native, t_end)
    train, valid, test = temporal_masks(
        frame, horizon_native, train_cut, val_cut)
    candidate_pool = list(ALL_FEATURES) + list(extra_candidates)

    metric_rows = []
    loo_rows = []
    coefficient_rows = []

    for label in LABELS:
        label_train, label_valid, label_test = label_population(
            frame, label, train, valid, test)
        final_train = label_train | label_valid

        if do_selection:
            selected, selection_rows = loo_ablation(
                frame,
                label,
                label_train,
                label_valid,
                candidate_pool,
                tolerance=prune_tol,
                max_train=selection_max_train,
                max_valid=selection_max_valid,
            )
        else:
            selected = list(candidate_pool)
            selection_rows = []

        for row in selection_rows:
            loo_rows.append({
                "dataset": dataset,
                "horizon": horizon_value,
                "unit": unit,
                "label": label,
                **row,
            })

        for row in standardized_logreg_coefficients(
            frame, label, final_train, candidate_pool
        ):
            coefficient_rows.append({
                "dataset": dataset,
                "horizon": horizon_value,
                "unit": unit,
                "label": label,
                **row,
            })

        sets = feature_sets(selected, extra_candidates)
        predictions = {}
        rows_by_key = {}
        prevalence = (
            float(frame[label].to_numpy()[label_test].mean())
            if label_test.sum() else float("nan")
        )

        print(
            f"\n  {label} | h={horizon_value:g}{unit} "
            f"| train={label_train.sum():,} "
            f"valid={label_valid.sum():,} test={label_test.sum():,} "
            f"| prevalence={prevalence:.6f}"
        )
        if do_selection:
            print(
                f"    selected {len(selected)}/{len(candidate_pool)}: "
                f"{', '.join(selected)}"
            )
            if extra_candidates:
                chosen = [c for c in extra_candidates if c in selected]
                print(
                    f"    candidates kept: "
                    f"{', '.join(chosen) if chosen else '(none)'}"
                )

        for model_name, model_fn in models:
            for set_name, features in sets.items():
                result = fit_predict(
                    frame,
                    label,
                    final_train,
                    label_test,
                    features,
                    model_fn,
                )
                roc, ap = prediction_scores(result)
                predictions[(model_name, set_name)] = result
                row = {
                    "dataset": dataset,
                    "horizon": horizon_value,
                    "unit": unit,
                    "label": label,
                    "model": model_name,
                    "feature_set": set_name,
                    "n_features": len(features),
                    "features": "|".join(features),
                    "n_train": int(label_train.sum()),
                    "n_validation": int(label_valid.sum()),
                    "n_test": int(label_test.sum()),
                    "test_positives": int(
                        frame[label].to_numpy()[label_test].sum()),
                    "test_prevalence": prevalence,
                    "roc_auc": roc,
                    "average_precision": ap,
                    "relative_ap": (
                        ap / prevalence
                        if prevalence > 0 and np.isfinite(ap)
                        else float("nan")
                    ),
                    "delta_ap_vs_graph": float("nan"),
                    "delta_ap_ci_low": float("nan"),
                    "delta_ap_ci_high": float("nan"),
                }
                metric_rows.append(row)
                rows_by_key[(model_name, set_name)] = row

            graph_result = predictions[(model_name, "GRAPH")]
            if graph_result is None:
                continue
            y_test, graph_probabilities = graph_result
            graph_ap = average_precision_score(
                y_test, graph_probabilities)
            for set_name in sets:
                if set_name == "GRAPH":
                    continue
                alternative = predictions[(model_name, set_name)]
                if alternative is None:
                    continue
                _, alternative_probabilities = alternative
                alternative_ap = average_precision_score(
                    y_test, alternative_probabilities)
                row = rows_by_key[(model_name, set_name)]
                row["delta_ap_vs_graph"] = (
                    alternative_ap - graph_ap)
                # Bootstrap only the nonlinear model, or LogReg if it is the
                # only available model.
                if model_name == models[-1][0]:
                    low, high = paired_event_bootstrap(
                        y_test,
                        graph_probabilities,
                        alternative_probabilities,
                        frame["eid"].to_numpy()[label_test],
                        n_boot=n_boot,
                        seed=sum(map(ord, label + set_name)),
                    )
                    row["delta_ap_ci_low"] = low
                    row["delta_ap_ci_high"] = high

        headline_model = models[-1][0]
        for set_name in sets:
            row = rows_by_key.get((headline_model, set_name))
            if row is None:
                continue
            delta = row["delta_ap_vs_graph"]
            delta_text = (
                f" dAP={delta:+.4f}"
                if np.isfinite(delta) else ""
            )
            print(
                f"    {headline_model:<8} {set_name:<25} "
                f"AP={row['average_precision']:.4f}{delta_text}"
            )

    return metric_rows, loo_rows, coefficient_rows


# ---------------------------------------------------------------------------
# Dataset runner and CLI

def run_dataset(name, args, output_dir):
    cfg = DATASETS[name]
    theta = (
        cfg["default_theta"] if args.theta is None else args.theta)
    horizons = (
        [float(value) for value in args.horizons.split(",")]
        if args.horizons
        else list(cfg["default_horizons"])
    )

    print(f">> loading {name}")
    events = load_dataset(
        name,
        limit=args.limit,
        max_size=args.max_size,
    )
    if not events:
        raise ValueError(f"No events loaded for {name}")
    print(f"   {len(events):,} events")

    print(f">> building causal rows once (theta={theta})")
    delay_frame = build_causal_rows(events, theta=theta)
    if args.min_recipients > 1:
        minimum_k = 1 + args.min_recipients
        keep = delay_frame["k"].to_numpy() >= minimum_k
        print(
            f"   focal population k>={minimum_k}: "
            f"{keep.sum():,}/{len(delay_frame):,} rows"
        )
        delay_frame = delay_frame[keep].reset_index(drop=True)
    if len(delay_frame) == 0:
        raise ValueError("No focal samples after filtering")

    train_cut, val_cut = temporal_cutoffs(
        delay_frame["t"].to_numpy(),
        train_q=args.train_q,
        val_q=args.val_q,
    )
    models = available_models()
    t_end = events[-1][2]

    selection_horizons = (
        {float(x) for x in args.selection_horizons.split(",") if x}
        if args.selection_horizons else None
    )

    metrics = []
    loo = []
    coefficients = []
    for horizon_value in horizons:
        horizon_native = horizon_value * cfg["step"]
        do_selection = not args.no_feature_selection and (
            selection_horizons is None
            or horizon_value in selection_horizons
        )
        result = evaluate_horizon(
            name,
            delay_frame,
            horizon_value,
            horizon_native,
            cfg["unit"],
            t_end,
            train_cut,
            val_cut,
            models,
            do_selection=do_selection,
            prune_tol=args.prune_tol,
            selection_max_train=args.selection_max_train,
            selection_max_valid=args.selection_max_valid,
            n_boot=args.n_boot,
            extra_candidates=[
                c for c in args.extra_candidates.split(",") if c
            ],
        )
        metrics.extend(result[0])
        loo.extend(result[1])
        coefficients.extend(result[2])

    pd.DataFrame(metrics, columns=METRIC_COLUMNS).to_csv(
        output_dir / f"{name}_metrics.csv", index=False)
    pd.DataFrame(loo, columns=LOO_COLUMNS).to_csv(
        output_dir / f"{name}_loo_validation.csv", index=False)
    pd.DataFrame(coefficients, columns=COEFFICIENT_COLUMNS).to_csv(
        output_dir / f"{name}_logreg_coefficients.csv", index=False)

    diagnostics = {
        "dataset": name,
        "theta": theta,
        "horizons": horizons,
        "unit": cfg["unit"],
        "n_events": len(events),
        "n_samples": len(delay_frame),
        "min_recipients": args.min_recipients,
        "max_size": args.max_size,
        "train_q": args.train_q,
        "val_q": args.val_q,
        "train_cut": train_cut,
        "validation_cut": val_cut,
        "graph_features": GRAPH_FEATURES,
        "hypergraph_features": HYPERGRAPH_FEATURES,
    }
    with open(output_dir / f"{name}_diagnostics.json", "w") as fh:
        json.dump(diagnostics, fh, indent=2)
    return metrics, loo, coefficients, diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS) + ["all"],
        default="emaileu",
    )
    parser.add_argument(
        "--horizons",
        default=None,
        help="comma-separated horizons in dataset units, e.g. 1,2,7,30",
    )
    parser.add_argument("--theta", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-size", type=int, default=1000)
    parser.add_argument(
        "--min-recipients",
        type=int,
        default=2,
        help="default 2 gives the shared focal population k>=3",
    )
    parser.add_argument("--train-q", type=float, default=0.50)
    parser.add_argument("--val-q", type=float, default=0.75)
    parser.add_argument("--prune-tol", type=float, default=0.001)
    parser.add_argument("--selection-max-train", type=int, default=250_000)
    parser.add_argument("--selection-max-valid", type=int, default=100_000)
    parser.add_argument("--no-feature-selection", action="store_true")
    parser.add_argument(
        "--selection-horizons",
        default=None,
        help="comma-separated horizons on which to run validation feature "
             "selection (others skip it); default = all horizons",
    )
    parser.add_argument(
        "--extra-candidates",
        default="",
        help="comma-separated candidate features to offer to validation "
             "selection (e.g. h_corecip_grouprecip,h_team_leadcov_s)",
    )
    parser.add_argument("--n-boot", type=int, default=200)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "link_prediction_results"),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = list(DATASETS) if args.dataset == "all" else [args.dataset]

    all_metrics = []
    all_loo = []
    all_coefficients = []
    all_diagnostics = []
    for name in names:
        metrics, loo, coefficients, diagnostics = run_dataset(
            name, args, output_dir)
        all_metrics.extend(metrics)
        all_loo.extend(loo)
        all_coefficients.extend(coefficients)
        all_diagnostics.append(diagnostics)

    if len(names) > 1:
        pd.DataFrame(all_metrics, columns=METRIC_COLUMNS).to_csv(
            output_dir / "all_metrics.csv", index=False)
        pd.DataFrame(all_loo, columns=LOO_COLUMNS).to_csv(
            output_dir / "all_loo_validation.csv", index=False)
        pd.DataFrame(
            all_coefficients, columns=COEFFICIENT_COLUMNS
        ).to_csv(
            output_dir / "all_logreg_coefficients.csv", index=False)
        with open(output_dir / "all_diagnostics.json", "w") as fh:
            json.dump(all_diagnostics, fh, indent=2)

    print(f"\n>> Results written to {output_dir}")


if __name__ == "__main__":
    main()
