"""Small, observation-only sampling profiles shared by both memory levels."""

from itertools import combinations
from statistics import median
from typing import Any

from .models import UNKNOWN, NOT_APPLICABLE


SAMPLING_FIELDS = ("sampling_interval", "sampling_regularity", "timestamp_alignment", "point_distribution")


def timestamp_samples(observations: dict[str, Any]) -> list[tuple[dict, set[int]]]:
    result = []
    for item in observations.get("measurements", []) or []:
        if not isinstance(item, dict):
            continue
        values = item.get("timestamps_ms")
        if isinstance(values, list):
            times = {int(t) for t in values if isinstance(t, (int, float)) and not isinstance(t, bool)}
            result.append((item, times))
    return result


def sampling_features(observations: dict[str, Any], bounds=None) -> dict[str, str]:
    """Describe a common sampled interval, never infer alignment from endpoints.

    Prefixes are truncated to the earliest last observation so different LIMIT
    cutoffs do not masquerade as timestamp misalignment. These are sampled
    features, not whole-database statistics.
    """
    result = dict.fromkeys(SAMPLING_FIELDS, UNKNOWN)
    samples = timestamp_samples(observations)
    if not samples or any(not times for _, times in samples):
        return result
    start = max(min(times) for _, times in samples)
    end = min(max(times) for _, times in samples)
    if bounds:
        start = max(start, bounds[0]) if bounds[0] is not None else start
        end = min(end, bounds[1]) if bounds[1] is not None else end
    if end <= start:
        return result
    series = [sorted(t for t in times if start <= t <= end) for _, times in samples]
    intervals = [[b - a for a, b in zip(times, times[1:])] for times in series]
    typical = [median(gaps) for gaps in intervals if gaps]
    if typical:
        value = median(typical)
        thresholds = ((1000, "subsecond"), (60000, "second"), (3600000, "minute"),
                      (86400000, "hour"), (float("inf"), "day_or_more"))
        result["sampling_interval"] = next(label for upper, label in thresholds if value < upper)
        regular = [max(gaps) <= 1.5 * median(gaps) for gaps in intervals if len(gaps) >= 2]
        if regular:
            result["sampling_regularity"] = "regular" if all(regular) else "irregular"
    if len(series) == 1:
        result["timestamp_alignment"] = NOT_APPLICABLE
    else:
        overlaps = [len(set(a) & set(b)) / min(len(a), len(b))
                    for a, b in combinations(series, 2) if a and b]
        if overlaps:
            overlap = min(overlaps)
            result["timestamp_alignment"] = "aligned" if overlap >= .9 else "partial" if overlap > 0 else "asynchronous"
    distributions = []
    for times in series:
        counts = [0] * 4
        for t in times:
            counts[min(3, int(4 * (t - start) / (end - start)))] += 1
        mean = sum(counts) / 4
        if mean and len(times) >= 4:
            distributions.append((sum((n - mean) ** 2 for n in counts) / 4) ** .5 / mean)
    if distributions:
        result["point_distribution"] = "uniform" if max(distributions) <= .5 else "uneven"
    return result


def inputs_have_common_timestamps(observations: dict[str, Any], names: list[str]) -> bool:
    """Reject observed asynchronous operands within a device; unknown is neutral."""
    by_device: dict[str, dict[str, set[int]]] = {}
    for item, times in timestamp_samples(observations):
        name = item.get("measurement") or item.get("name")
        if name in names:
            by_device.setdefault(item.get("device", ""), {})[name] = times
    for series in by_device.values():
        if not all(name in series and series[name] for name in names):
            continue
        start = max(min(series[name]) for name in names)
        end = min(max(series[name]) for name in names)
        if start <= end and not set.intersection(*(series[name] for name in names)):
            return False
        if start > end:
            return False
    return True
