"""Launch-day Twitch viewer forecaster for gta6-twitch-ccu-forecast-01.

Reads one case (schema "makefun/twitch-ccu-case/v1") per *.in file in the input
directory and writes, for each of hours 2..24 of the release day, five quantiles
(0.05, 0.25, 0.5, 0.75, 0.95) of that hour's mean concurrent viewers in the
game's Twitch category.

Method
------
The case itself ships the calibration data: `comparables`, dozens of past big
game launches with hourly category viewers from 14 days before to 3 days after
their release day. For every comparable we take its release-day curve and form,
for each hour h, the log ratio ln(viewers_h / viewers_1) against its own hour-1
count. A launch's day shape depends strongly on how big its first hour was:
a rolling midnight console launch whose hour 1 already holds the European
midnight crowd behaves nothing like a simultaneous global unlock whose hour 1
is still a trickle. So for each hour we fit a *local-linear* model of the log
ratio on the comparable's log hour-1 size, with Gaussian kernel weights
(bandwidth 2.0 in log space) centred on this case's own hour-1 print, and an
extra weight (x1.8) for console-only launches, the closest structural analogues
of a console-first release. The median forecast is the fitted line's value at
the case's size; the other quantiles are the fitted value plus the matching
weighted quantiles of the fit's residuals, which widens the spread exactly
where comparable launches disagreed most. Hours with too few usable
comparables, and cases with no usable comparables at all, fall back to the
persistence curve (flat at hour 1, log-normal spread growing like sqrt(h-1)).

Everything is deterministic, standard library only, and runs in well under a
second per case. Quantiles are clamped above 0, capped at 1e9, and forced
non-decreasing, as the output contract requires.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first (harmless here,
# this entry is a single file, but keeps behaviour identical to the starter).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]
Z_BY_LEVEL = dict(zip(LEVELS, Z))
FALLBACK_SIGMA = 0.15  # persistence fallback spread, per sqrt(hour - 1)

BW = 2.0               # kernel bandwidth on log hour-1 size
CONSOLE_BOOST = 1.8    # extra kernel weight for launches with no PC version
RIDGE = 0.05           # variance floor for the local-linear slope
MIN_SAMPLES = 6        # usable (hour, comparable) points before trusting a fit


def parse_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def day_curve(comp):
    """Release-day hourly viewers of one comparable: {hour 1..24: viewers}."""
    try:
        start = parse_ts(comp["day_start"])
    except Exception:
        return {}
    curve = {}
    for point in comp.get("hourly") or []:
        viewers = point.get("viewers")
        if viewers is None:
            continue
        try:
            delta = (parse_ts(point["from"]) - start).total_seconds()
        except Exception:
            continue
        if 0 <= delta < 24 * 3600:
            curve[int(delta // 3600) + 1] = viewers
    return curve


def weighted_quantile(pairs, tau):
    """Quantile of (value, weight) pairs, linear interpolation on the CDF."""
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    if total <= 0:
        return pairs[len(pairs) // 2][0]
    target = tau * total
    acc = 0.0
    for i, (value, weight) in enumerate(pairs):
        acc += weight
        if acc >= target:
            if i == 0:
                return value
            prev_value = pairs[i - 1][0]
            frac = (target - (acc - weight)) / weight if weight > 0 else 0.0
            return prev_value + frac * (value - prev_value)
    return pairs[-1][0]


def round2(x):
    """Round to 2 decimals, half up; whole numbers come back as int."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if float(value).is_integer() else value


def fallback_quantiles(p1, h, taus):
    return [p1 * math.exp(z * FALLBACK_SIGMA * math.sqrt(h - 1))
            for z in (normal_quantile(t) for t in taus)]


def normal_quantile(tau):
    """Standard normal quantile (Acklam's rational approximation)."""
    if tau in Z_BY_LEVEL:
        return Z_BY_LEVEL[tau]
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if tau < plow:
        q = math.sqrt(-2 * math.log(tau))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if tau > phigh:
        q = math.sqrt(-2 * math.log(1 - tau))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = tau - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def hour_quantiles(p1, log_p1, pool, h, taus):
    """The quantiles for one hour from the comparable pool, at levels `taus`.

    pool entries: (log hour-1 size, kernel weight, day curve).
    """
    samples = []
    for log_v1, weight, curve in pool:
        v1 = curve.get(1)
        vh = curve.get(h)
        if v1 and vh and v1 > 0 and vh > 0:
            samples.append((log_v1, math.log(vh / v1), weight))
    if len(samples) < MIN_SAMPLES:
        return fallback_quantiles(p1, h, taus)

    sw = sum(w for _, _, w in samples)
    mx = sum(w * x for x, _, w in samples) / sw
    my = sum(w * y for _, y, w in samples) / sw
    sxx = sum(w * (x - mx) ** 2 for x, _, w in samples) / sw + RIDGE
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in samples) / sw
    beta = sxy / sxx
    mu = my + beta * (log_p1 - mx)
    residuals = [(y - (my + beta * (x - mx)), w) for x, y, w in samples]

    quantiles = []
    for tau in taus:
        if tau == 0.5:
            log_q = mu
        else:
            log_q = mu + weighted_quantile(residuals, tau)
        quantiles.append(p1 * math.exp(log_q))
    return quantiles


def clean(quantiles):
    out = []
    prev = 0.0
    for q in quantiles:
        q = min(max(q, 0.01), 1_000_000_000.0)
        q = max(q, prev)
        out.append(round2(q))
        prev = out[-1]
    return out


def forecast_case(case):
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    p1 = 1.0
    if first is not None and first.get("viewers"):
        p1 = max(1.0, float(first["viewers"]))
    log_p1 = math.log(p1)

    pool = []
    for comp in case.get("comparables") or []:
        curve = day_curve(comp)
        v1 = curve.get(1)
        if not v1 or v1 <= 0:
            continue
        log_v1 = math.log(v1)
        d = log_v1 - log_p1
        weight = math.exp(-(d * d) / (2 * BW * BW))
        platforms = (comp.get("platforms") or "").lower()
        if "pc" not in platforms:
            weight *= CONSOLE_BOOST
        pool.append((log_v1, weight, curve))

    taus = case.get("levels")
    if not (isinstance(taus, list) and taus
            and all(isinstance(t, (int, float)) and 0 < t < 1 for t in taus)):
        taus = LEVELS

    rows = []
    for row in case.get("forecast") or []:
        h = row["hour"]
        rows.append({"hour": h, "q": clean(hour_quantiles(p1, log_p1, pool, h, taus))})
    if not rows:  # a case with no forecast rows still gets a full, valid answer
        rows = [{"hour": h, "q": clean(fallback_quantiles(p1, h, taus))} for h in range(2, 25)]
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
