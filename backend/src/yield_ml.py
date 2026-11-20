"""Mould-grouped good-rate lower bounds.

Only signed (closed) actuals count as training data: their good/total shot
figures are frozen once the cost review is signed.  Validation is a
leave-one-out check performed inside the mould group, so the bound is
calibrated on the same mould it will later promise against.
"""
import math
from collections import defaultdict
from sqlalchemy import select
from .models import Mould, Estimate
from .execution_models import TrialExecution, CostReview

MIN_GROUP_SAMPLES = 5
DEFAULT_Z = 1.645
FLOOR_CLIP = 0.05


def signed_rates(s):
    """Return {mould_id: [(execution_id, good_shots, total_shots, rate), ...]}
    for settled executions whose cost review has been signed."""
    rows = s.execute(
        select(TrialExecution, Mould.id)
        .join(CostReview, CostReview.execution_id == TrialExecution.id)
        .join(Estimate, TrialExecution.estimate_id == Estimate.id)
        .join(Mould, Estimate.mould_id == Mould.id)
        .where(CostReview.status == 'signed',
               TrialExecution.status == 'settled',
               TrialExecution.total_shots > 0)
    ).all()
    groups = defaultdict(list)
    for x, mould_id in rows:
        groups[mould_id].append((x.id, x.good_shots, x.total_shots,
                                 x.good_shots / x.total_shots))
    return groups


def _mean_std(values):
    n = len(values)
    mean = sum(values) / n
    # Population standard deviation; with tiny groups the sample estimator
    # would inflate confidence half-widths dramatically.
    var = sum((v - mean) ** 2 for v in values) / n
    return mean, math.sqrt(var)


def lower_bound(rates, z=DEFAULT_Z):
    """One-sided conservative lower rate: mean - z*std, clipped to (0, 1]."""
    mean, std = _mean_std(rates)
    bound = mean - z * std
    return min(1.0, max(FLOOR_CLIP, bound)), mean, std


def loo_coverage(rates, z=DEFAULT_Z):
    """Leave-one-out: does the held-out rate stay above the bound trained
    on the remaining n-1 points?  Returns (covered, total, per-fold rows)."""
    folds = []
    covered = 0
    for i, held in enumerate(rates):
        rest = rates[:i] + rates[i + 1:]
        bound, _, _ = lower_bound(rest, z)
        ok = held >= bound
        covered += 1 if ok else 0
        folds.append({'held_out_rate': round(held, 4),
                      'loo_lower_rate': round(bound, 4),
                      'covered': ok})
    return covered, len(rates), folds


def train_group(mould_id, points, z=DEFAULT_Z):
    """Build a training report for one mould group.

    Returns (report | None, reason).  Groups below MIN_GROUP_SAMPLES are
    refused — their orders must stay suspended until more signed actuals."""
    rates = [p[3] for p in points]
    if len(rates) < MIN_GROUP_SAMPLES:
        return None, f'模具已签实绩仅 {len(rates)} 条，少于 {MIN_GROUP_SAMPLES} 条，良品率下界暂不可训'
    bound, mean, std = lower_bound(rates, z)
    covered, total, folds = loo_coverage(rates, z)
    report = {
        'mould_id': mould_id,
        'lower_rate': round(bound, 4),
        'sample_size': len(rates),
        'mean_rate': round(mean, 4),
        'std_rate': round(std, 4),
        'z': z,
        'fold_rows': [{'execution_id': p[0], 'good_shots': p[1],
                       'total_shots': p[2], **f}
                      for p, f in zip(points, folds)],
        'metrics': {
            'loo_covered': covered,
            'loo_total': total,
            'loo_coverage': round(covered / total, 4),
            'min_observed_rate': round(min(rates), 4),
            'max_observed_rate': round(max(rates), 4),
        },
    }
    return report, None


def train_all(s, only_mould_id=None, z=DEFAULT_Z):
    groups = signed_rates(s)
    reports, failures = [], []
    mould_ids = [only_mould_id] if only_mould_id else sorted(groups)
    for mould_id in mould_ids:
        if only_mould_id and not s.get(Mould, mould_id):
            failures.append({'mould_id': mould_id, 'reason': '模具不存在'})
            continue
        points = groups.get(mould_id, [])
        report, reason = train_group(mould_id, points, z)
        if report:
            reports.append(report)
        else:
            failures.append({'mould_id': mould_id, 'reason': reason})
    return reports, failures
