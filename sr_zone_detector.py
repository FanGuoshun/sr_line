import argparse
import json
import math
import os
import ssl
import statistics
import subprocess
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


@dataclass
class Candle:
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Pivot:
    index: int
    price: float
    kind: str
    strength: float


@dataclass
class Zone:
    kind: str
    lower: float
    upper: float
    touches: int
    strength_sum: float
    latest_index: int

    @property
    def mid(self) -> float:
        return (self.lower + self.upper) / 2.0

    @property
    def width_pct(self) -> float:
        if self.mid <= 0:
            return 0.0
        return (self.upper - self.lower) / self.mid * 100.0


@dataclass
class Trendline:
    kind: str
    anchor_index: int
    anchor_price: float
    slope: float
    start_index: int
    end_index: int
    touches: int
    score: float


def fetch_binance_futures_klines(symbol: str, interval: str, limit: int) -> List[Candle]:
    query = urllib.parse.urlencode({"symbol": symbol, "interval": interval, "limit": limit})
    hosts = [
        "https://fapi.binance.com",
        "https://fapi1.binance.com",
        "https://fapi2.binance.com",
        "https://fapi3.binance.com",
        "https://fapi4.binance.com",
    ]
    urls = [f"{host}/fapi/v1/klines?{query}" for host in hosts]
    data = None
    for url in urls:
        req = urllib.request.Request(url, headers={"User-Agent": "sr-zone-detector/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                break
        except Exception:
            pass
        try:
            insecure_ctx = ssl._create_unverified_context()
            with urllib.request.urlopen(req, timeout=20, context=insecure_ctx) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                break
        except Exception:
            pass
        for curl_cmd in [
            [
                "curl",
                "-sS",
                "-L",
                "--http1.1",
                "--retry",
                "4",
                "--retry-all-errors",
                "--retry-delay",
                "1",
                "--max-time",
                "30",
                url,
            ],
            [
                "curl",
                "-k",
                "-sS",
                "-L",
                "--http1.1",
                "--retry",
                "4",
                "--retry-all-errors",
                "--retry-delay",
                "1",
                "--max-time",
                "30",
                url,
            ],
        ]:
            try:
                completed = subprocess.run(
                    curl_cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                data = json.loads(completed.stdout)
                break
            except Exception:
                continue
        if data is not None:
            break
    if data is None:
        raise RuntimeError("Failed to fetch klines from Binance futures API")
    candles = []
    for row in data:
        candles.append(
            Candle(
                open_time=int(row[0]),
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
            )
        )
    return candles


def compute_true_ranges(candles: List[Candle]) -> List[float]:
    tr = []
    for i, c in enumerate(candles):
        if i == 0:
            tr.append(c.high - c.low)
            continue
        prev_close = candles[i - 1].close
        tr.append(max(c.high - c.low, abs(c.high - prev_close), abs(c.low - prev_close)))
    return tr


def compute_atr(candles: List[Candle], period: int = 14) -> float:
    tr = compute_true_ranges(candles)
    if not tr:
        return 0.0
    if len(tr) < period:
        return statistics.mean(tr)
    return statistics.mean(tr[-period:])


def detect_pivots(candles: List[Candle], lookback: int = 3) -> List[Pivot]:
    pivots: List[Pivot] = []
    if len(candles) < lookback * 2 + 1:
        return pivots
    for i in range(lookback, len(candles) - lookback):
        center = candles[i]
        left = candles[i - lookback : i]
        right = candles[i + 1 : i + 1 + lookback]
        local_high = max([x.high for x in left + [center] + right])
        local_low = min([x.low for x in left + [center] + right])
        if math.isclose(center.high, local_high, rel_tol=1e-12) or center.high >= local_high:
            strength = (center.high - max(x.close for x in left)) + (center.high - max(x.close for x in right))
            pivots.append(Pivot(index=i, price=center.high, kind="resistance", strength=max(0.0, strength)))
        if math.isclose(center.low, local_low, rel_tol=1e-12) or center.low <= local_low:
            strength = (min(x.close for x in left) - center.low) + (min(x.close for x in right) - center.low)
            pivots.append(Pivot(index=i, price=center.low, kind="support", strength=max(0.0, strength)))
    return pivots


def build_zones(
    pivots: List[Pivot],
    current_price: float,
    atr: float,
    base_min_zone_pct: float = 0.35,
    atr_zone_mult: float = 0.35,
) -> List[Zone]:
    zones: List[Zone] = []
    base_half_width = max(current_price * base_min_zone_pct / 100.0, atr * atr_zone_mult)
    for p in sorted(pivots, key=lambda x: x.price):
        matched = None
        for z in zones:
            if z.kind != p.kind:
                continue
            if abs(p.price - z.mid) <= base_half_width:
                matched = z
                break
        if matched is None:
            zones.append(
                Zone(
                    kind=p.kind,
                    lower=p.price - base_half_width,
                    upper=p.price + base_half_width,
                    touches=1,
                    strength_sum=p.strength,
                    latest_index=p.index,
                )
            )
        else:
            matched.lower = min(matched.lower, p.price - base_half_width)
            matched.upper = max(matched.upper, p.price + base_half_width)
            matched.touches += 1
            matched.strength_sum += p.strength
            matched.latest_index = max(matched.latest_index, p.index)
    return zones


def score_zone(zone: Zone, latest_index: int, current_price: float) -> float:
    recency = 1.0 / (1.0 + max(0, latest_index - zone.latest_index))
    distance = abs(zone.mid - current_price) / current_price if current_price > 0 else 1.0
    distance_factor = 1.0 / (1.0 + distance * 10.0)
    touch_factor = zone.touches
    strength_factor = math.log1p(zone.strength_sum) if zone.strength_sum > 0 else 0.0
    return touch_factor * 1.8 + strength_factor * 1.2 + recency * 2.0 + distance_factor * 1.0


def format_ts(ms: int) -> str:
    dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M UTC")


def trendline_price(line: Trendline, x_index: int) -> float:
    return line.anchor_price + line.slope * (x_index - line.anchor_index)


def detect_trendlines(
    candles: List[Candle],
    pivots: List[Pivot],
    atr: float,
    top_n: int,
    min_touches: int = 3,
    min_span: int = 24,
    tol_atr_mult: float = 0.25,
    window_size: int = 100,
) -> List[Trendline]:
    if len(candles) < 40 or window_size <= 0:
        return []
    tolerance = max(atr * tol_atr_mult, candles[-1].close * 0.0015)
    n = len(candles)
    candidates: List[Trendline] = []
    for window_start in range(0, n, window_size):
        window_end = min(window_start + window_size, n)
        if window_end - window_start < 40:
            continue
        for kind, slope_check in [("support", lambda s: s > 0), ("resistance", lambda s: s < 0)]:
            points = sorted(
                [p for p in pivots if p.kind == kind and window_start <= p.index < window_end],
                key=lambda p: p.index,
            )
            if len(points) < 3:
                continue
            for i in range(len(points) - 1):
                p1 = points[i]
                for j in range(i + 1, len(points)):
                    p2 = points[j]
                    span = p2.index - p1.index
                    if span < min_span:
                        continue
                    slope = (p2.price - p1.price) / span
                    if not slope_check(slope):
                        continue
                    touches = 0
                    for p in points:
                        if p.index < p1.index:
                            continue
                        expected = p1.price + slope * (p.index - p1.index)
                        if abs(p.price - expected) <= tolerance:
                            touches += 1
                    if touches < min_touches:
                        continue
                    violations = 0
                    for k in range(p1.index, window_end):
                        expected = p1.price + slope * (k - p1.index)
                        if kind == "support" and candles[k].low < expected - tolerance * 1.2:
                            violations += 1
                        if kind == "resistance" and candles[k].high > expected + tolerance * 1.2:
                            violations += 1
                    max_violations = max(2, int(0.08 * (window_end - p1.index)))
                    if violations > max_violations:
                        continue
                    recency = 1.0 / (1.0 + (window_end - 1 - p2.index))
                    score = touches * 2.4 + span / 36.0 + recency * 2.0 - violations * 0.7
                    candidates.append(
                        Trendline(
                            kind=kind,
                            anchor_index=p1.index,
                            anchor_price=p1.price,
                            slope=slope,
                            start_index=p1.index,
                            end_index=window_end - 1,
                            touches=touches,
                            score=score,
                        )
                    )
    selected: List[Trendline] = []
    for kind in ["support", "resistance"]:
        subset = sorted([x for x in candidates if x.kind == kind], key=lambda x: x.score, reverse=True)
        kept: List[Trendline] = []
        for line in subset:
            too_similar = False
            for k in kept:
                slope_close = abs(line.slope - k.slope) <= max(abs(k.slope) * 0.25, 1e-9)
                price_close = abs(line.anchor_price - k.anchor_price) <= tolerance * 1.6
                if slope_close and price_close:
                    too_similar = True
                    break
            if not too_similar:
                kept.append(line)
            if len(kept) >= top_n:
                break
        selected.extend(kept)
    return selected


def plot_candles_with_zones(
    candles: List[Candle],
    supports: List[Zone],
    resistances: List[Zone],
    trendlines: List[Trendline],
    top_n: int,
    output_path: str,
    symbol: str,
    interval: str,
    bars: int,
):
    if bars <= 0:
        bars = len(candles)
    view = candles[-bars:] if bars < len(candles) else candles
    fig, ax = plt.subplots(figsize=(16, 8))
    x_values = list(range(len(view)))
    width = 0.62
    for x, c in zip(x_values, view):
        color = "#26a69a" if c.close >= c.open else "#ef5350"
        ax.vlines(x, c.low, c.high, color=color, linewidth=1.0, alpha=0.9, zorder=2)
        body_bottom = min(c.open, c.close)
        body_height = max(abs(c.close - c.open), 0.01)
        rect = Rectangle(
            (x - width / 2, body_bottom),
            width,
            body_height,
            facecolor=color,
            edgecolor=color,
            linewidth=1.0,
            alpha=0.9,
            zorder=3,
        )
        ax.add_patch(rect)
    xmin = -1
    xmax = len(view)
    view_start = len(candles) - len(view)
    view_end = len(candles) - 1
    for z in supports[:top_n]:
        ax.axhspan(z.lower, z.upper, xmin=0.0, xmax=1.0, facecolor="#2e7d32", alpha=0.18, edgecolor="none", zorder=1)
    for z in resistances[:top_n]:
        ax.axhspan(z.lower, z.upper, xmin=0.0, xmax=1.0, facecolor="#c62828", alpha=0.18, edgecolor="none", zorder=1)
    for line in trendlines:
        x0_global = max(view_start, line.start_index)
        x1_global = min(view_end, line.end_index)
        if x1_global <= x0_global:
            continue
        y0 = trendline_price(line, x0_global)
        y1 = trendline_price(line, x1_global)
        x0 = x0_global - view_start
        x1 = x1_global - view_start
        color = "#1b5e20" if line.kind == "support" else "#b71c1c"
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=2.0, alpha=0.9, zorder=4)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("Bar Index")
    ax.set_ylabel("Price")
    ax.set_title(f"{symbol} {interval} Candles with SR Zones")
    ax.grid(alpha=0.2, linestyle="--")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--interval", default="4h")
    parser.add_argument("--limit", type=int, default=800)
    parser.add_argument("--lookback", type=int, default=4)
    parser.add_argument("--top", type=int, default=4)
    parser.add_argument("--min-zone-pct", type=float, default=0.35)
    parser.add_argument("--atr-zone-mult", type=float, default=0.35)
    parser.add_argument("--plot-file", default="sr_zones.png")
    parser.add_argument("--plot-bars", type=int, default=200)
    parser.add_argument("--trend-top", type=int, default=2)
    parser.add_argument("--trend-min-touches", type=int, default=3)
    parser.add_argument("--trend-min-span", type=int, default=24)
    parser.add_argument("--trend-window", type=int, default=100)
    args = parser.parse_args()

    candles = fetch_binance_futures_klines(args.symbol, args.interval, args.limit)
    if not candles:
        raise RuntimeError("No kline data fetched")

    current_price = candles[-1].close
    atr = compute_atr(candles, period=14)
    pivots = detect_pivots(candles, lookback=args.lookback)
    zones = build_zones(
        pivots=pivots,
        current_price=current_price,
        atr=atr,
        base_min_zone_pct=args.min_zone_pct,
        atr_zone_mult=args.atr_zone_mult,
    )
    latest_index = len(candles) - 1
    ranked = sorted(
        zones,
        key=lambda z: score_zone(z, latest_index=latest_index, current_price=current_price),
        reverse=True,
    )

    supports = [z for z in ranked if z.kind == "support" and z.upper <= current_price * 1.02]
    resistances = [z for z in ranked if z.kind == "resistance" and z.lower >= current_price * 0.98]
    if len(supports) < args.top:
        supports = sorted([z for z in ranked if z.kind == "support"], key=lambda x: abs(x.mid - current_price))
    if len(resistances) < args.top:
        resistances = sorted([z for z in ranked if z.kind == "resistance"], key=lambda x: abs(x.mid - current_price))
    trendlines = detect_trendlines(
        candles=candles,
        pivots=pivots,
        atr=atr,
        top_n=args.trend_top,
        min_touches=args.trend_min_touches,
        min_span=args.trend_min_span,
        window_size=args.trend_window,
    )

    print(f"Symbol: {args.symbol} | Interval: {args.interval} | Bars: {len(candles)}")
    print(f"Last Candle Time: {format_ts(candles[-1].open_time)}")
    print(f"Current Price: {current_price:.4f}")
    print(f"ATR(14): {atr:.4f}")
    print(f"Pivot Count: {len(pivots)}")
    print()
    print("Key Support Zones:")
    for i, z in enumerate(supports[: args.top], 1):
        print(
            f"{i}. [{z.lower:.4f}, {z.upper:.4f}] "
            f"mid={z.mid:.4f} width={z.width_pct:.2f}% touches={z.touches}"
        )
    print()
    print("Key Resistance Zones:")
    for i, z in enumerate(resistances[: args.top], 1):
        print(
            f"{i}. [{z.lower:.4f}, {z.upper:.4f}] "
            f"mid={z.mid:.4f} width={z.width_pct:.2f}% touches={z.touches}"
        )
    print()
    print("Key Trendlines:")
    if not trendlines:
        print("None")
    for i, line in enumerate(sorted(trendlines, key=lambda x: x.score, reverse=True), 1):
        end_price = trendline_price(line, line.end_index)
        print(
            f"{i}. {line.kind} slope={line.slope:.4f} touches={line.touches} "
            f"start={line.start_index} end={line.end_index} last_price={end_price:.4f}"
        )
    if args.plot_file:
        plot_candles_with_zones(
            candles=candles,
            supports=supports,
            resistances=resistances,
            trendlines=trendlines,
            top_n=args.top,
            output_path=args.plot_file,
            symbol=args.symbol,
            interval=args.interval,
            bars=args.plot_bars,
        )
        print()
        print(f"Chart saved: {args.plot_file}")


if __name__ == "__main__":
    main()
