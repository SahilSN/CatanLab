from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


def wilson_interval(
    wins: int,
    games: int,
    z: float = 1.96,
) -> tuple[float, float]:
    """
    95% Wilson confidence interval for a binomial
    proportion.
    """
    if games == 0:
        return float("nan"), float("nan")

    p = wins / games
    z2 = z * z

    center = (
        p
        + z2 / (2 * games)
    ) / (
        1 + z2 / games
    )

    half = (
        z
        * math.sqrt(
            p * (1 - p) / games
            + z2 / (4 * games * games)
        )
        / (
            1 + z2 / games
        )
    )

    return center - half, center + half


def mean_ci(
    values: pd.Series,
    z: float = 1.96,
) -> tuple[float, float]:
    """
    Normal-approximation confidence interval for a
    sample mean.
    """
    values = values.dropna()

    n = len(values)

    if n == 0:
        return float("nan"), float("nan")

    mean = values.mean()

    if n == 1:
        return mean, mean

    std = values.std(ddof=1)
    se = std / math.sqrt(n)

    return (
        mean - z * se,
        mean + z * se,
    )


def strategy_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for strategy, group in df.groupby(
        "strategy",
        sort=True,
    ):
        games = len(group)

        if "won" in group.columns:
            wins = int(group["won"].sum())
        elif "is_winner" in group.columns:
            wins = int(
                group["is_winner"].sum()
            )
        else:
            raise KeyError(
                "Expected a 'won' or "
                "'is_winner' column."
            )

        win_low, win_high = wilson_interval(
            wins,
            games,
        )

        vp_low, vp_high = mean_ci(
            group["final_vp"]
        )

        rows.append({
            "strategy": strategy,
            "games": games,
            "wins": wins,
            "win_rate": wins / games,
            "win_ci_low": win_low,
            "win_ci_high": win_high,
            "avg_final_vp":
                group["final_vp"].mean(),
            "vp_ci_low": vp_low,
            "vp_ci_high": vp_high,
        })

    return pd.DataFrame(rows)


def seat_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    seat_col = None

    for candidate in (
        "seat",
        "seat_index",
        "player_id",
    ):
        if candidate in df.columns:
            seat_col = candidate
            break

    if seat_col is None:
        raise KeyError(
            "Could not find a seat column."
        )

    win_col = (
        "won"
        if "won" in df.columns
        else "is_winner"
    )

    return (
        df.groupby(
            seat_col,
            sort=True,
        )
        .agg(
            games=(
                win_col,
                "size",
            ),
            wins=(
                win_col,
                "sum",
            ),
            win_rate=(
                win_col,
                "mean",
            ),
            avg_final_vp=(
                "final_vp",
                "mean",
            ),
        )
        .reset_index()
        .rename(
            columns={
                seat_col: "seat"
            }
        )
    )


def strategy_by_seat(
    df: pd.DataFrame,
) -> pd.DataFrame:
    seat_col = None

    for candidate in (
        "seat",
        "seat_index",
        "player_id",
    ):
        if candidate in df.columns:
            seat_col = candidate
            break

    if seat_col is None:
        raise KeyError(
            "Could not find a seat column."
        )

    win_col = (
        "won"
        if "won" in df.columns
        else "is_winner"
    )

    return (
        df.groupby(
            [
                "strategy",
                seat_col,
            ],
            sort=True,
        )
        .agg(
            games=(
                win_col,
                "size",
            ),
            wins=(
                win_col,
                "sum",
            ),
            win_rate=(
                win_col,
                "mean",
            ),
            avg_final_vp=(
                "final_vp",
                "mean",
            ),
        )
        .reset_index()
        .rename(
            columns={
                seat_col: "seat"
            }
        )
    )


def lineup_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    lineup_col = None

    for candidate in (
        "lineup",
        "lineup_id",
        "strategy_lineup",
    ):
        if candidate in df.columns:
            lineup_col = candidate
            break

    if lineup_col is None:
        return pd.DataFrame()

    win_col = (
        "won"
        if "won" in df.columns
        else "is_winner"
    )

    return (
        df.groupby(
            [
                lineup_col,
                "strategy",
            ],
            sort=True,
        )
        .agg(
            games=(
                win_col,
                "size",
            ),
            wins=(
                win_col,
                "sum",
            ),
            win_rate=(
                win_col,
                "mean",
            ),
            avg_final_vp=(
                "final_vp",
                "mean",
            ),
        )
        .reset_index()
        .rename(
            columns={
                lineup_col: "lineup"
            }
        )
    )


def correlation_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    candidates = [
        "final_vp",
        "roads_built",
        "settlements_built",
        "cities_built",
        "dev_cards_bought",
        "maritime_trades",
        "domestic_trades",
        "has_longest_road",
        "has_largest_army",
        "turns_played",
    ]

    cols = [
        col
        for col in candidates
        if col in df.columns
    ]

    if "final_vp" not in cols:
        return pd.DataFrame()

    rows = []

    for strategy, group in df.groupby(
        "strategy",
        sort=True,
    ):
        numeric = group[
            cols
        ].apply(
            pd.to_numeric,
            errors="coerce",
        )

        corr = numeric.corr()

        for col in cols:
            if col == "final_vp":
                continue

            rows.append({
                "strategy": strategy,
                "metric": col,
                "corr_with_final_vp":
                    corr.loc[
                        "final_vp",
                        col,
                    ],
            })

    return pd.DataFrame(rows)


def print_strategy_table(
    summary: pd.DataFrame,
) -> None:
    print()
    print(
        "Strategy confidence intervals"
    )
    print("=" * 88)

    print(
        f"{'strategy':20s}"
        f"{'games':>7s}"
        f"{'win%':>9s}"
        f"{'95% CI':>19s}"
        f"{'VP':>8s}"
        f"{'95% VP CI':>21s}"
    )

    print("-" * 88)

    for _, row in summary.iterrows():
        print(
            f"{row['strategy']:20s}"
            f"{int(row['games']):7d}"
            f"{100 * row['win_rate']:8.2f}%"
            f"  "
            f"[{100 * row['win_ci_low']:5.2f}, "
            f"{100 * row['win_ci_high']:5.2f}]"
            f"{row['avg_final_vp']:8.3f}"
            f"  "
            f"[{row['vp_ci_low']:6.3f}, "
            f"{row['vp_ci_high']:6.3f}]"
        )



def pairwise_strategy_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare strategies using their aggregate win
    rates and final VP.

    Because every strategy has equal seat exposure,
    these differences are already seat-balanced by
    design.
    """
    win_col = (
        "won"
        if "won" in df.columns
        else "is_winner"
    )

    strategies = sorted(
        df["strategy"].unique()
    )

    rows = []

    for i, strategy_a in enumerate(strategies):
        a = df[
            df["strategy"] == strategy_a
        ]

        for strategy_b in strategies[i + 1:]:
            b = df[
                df["strategy"] == strategy_b
            ]

            p_a = a[win_col].mean()
            p_b = b[win_col].mean()

            n_a = len(a)
            n_b = len(b)

            se = math.sqrt(
                p_a * (1 - p_a) / n_a
                + p_b * (1 - p_b) / n_b
            )

            diff = p_a - p_b

            win_low = diff - 1.96 * se
            win_high = diff + 1.96 * se

            vp_a = a["final_vp"]
            vp_b = b["final_vp"]

            vp_diff = (
                vp_a.mean()
                - vp_b.mean()
            )

            vp_se = math.sqrt(
                vp_a.var(ddof=1) / len(vp_a)
                + vp_b.var(ddof=1) / len(vp_b)
            )

            vp_low = (
                vp_diff
                - 1.96 * vp_se
            )
            vp_high = (
                vp_diff
                + 1.96 * vp_se
            )

            rows.append({
                "strategy_a": strategy_a,
                "strategy_b": strategy_b,
                "win_rate_diff":
                    diff,
                "win_diff_ci_low":
                    win_low,
                "win_diff_ci_high":
                    win_high,
                "vp_diff":
                    vp_diff,
                "vp_diff_ci_low":
                    vp_low,
                "vp_diff_ci_high":
                    vp_high,
            })

    return pd.DataFrame(rows)


def lineup_sensitivity(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize how much each strategy's performance
    changes across opponent lineups.
    """
    lineup_col = None

    for candidate in (
        "lineup",
        "lineup_id",
        "strategy_lineup",
    ):
        if candidate in df.columns:
            lineup_col = candidate
            break

    if lineup_col is None:
        return pd.DataFrame()

    win_col = (
        "won"
        if "won" in df.columns
        else "is_winner"
    )

    per_lineup = (
        df.groupby(
            [
                "strategy",
                lineup_col,
            ],
            sort=True,
        )
        .agg(
            games=(
                win_col,
                "size",
            ),
            win_rate=(
                win_col,
                "mean",
            ),
            avg_final_vp=(
                "final_vp",
                "mean",
            ),
        )
        .reset_index()
    )

    rows = []

    for strategy, group in per_lineup.groupby(
        "strategy",
        sort=True,
    ):
        rows.append({
            "strategy": strategy,
            "lineups":
                len(group),
            "min_win_rate":
                group["win_rate"].min(),
            "max_win_rate":
                group["win_rate"].max(),
            "win_rate_range":
                (
                    group["win_rate"].max()
                    - group["win_rate"].min()
                ),
            "std_lineup_win_rate":
                group["win_rate"].std(ddof=1),
            "min_avg_vp":
                group["avg_final_vp"].min(),
            "max_avg_vp":
                group["avg_final_vp"].max(),
            "vp_range":
                (
                    group["avg_final_vp"].max()
                    - group["avg_final_vp"].min()
                ),
        })

    return pd.DataFrame(rows)

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "player_games_csv",
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    df = pd.read_csv(
        args.player_games_csv
    )

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else args.player_games_csv.parent
        / "analysis"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    strategy = strategy_summary(df)
    seat = seat_summary(df)
    by_seat = strategy_by_seat(df)
    lineup = lineup_summary(df)
    corr = correlation_summary(df)
    pairwise = pairwise_strategy_summary(df)
    sensitivity = lineup_sensitivity(df)

    strategy.to_csv(
        output_dir
        / "strategy_confidence_intervals.csv",
        index=False,
    )

    seat.to_csv(
        output_dir
        / "seat_summary.csv",
        index=False,
    )

    by_seat.to_csv(
        output_dir
        / "strategy_by_seat.csv",
        index=False,
    )

    if not lineup.empty:
        lineup.to_csv(
            output_dir
            / "strategy_by_lineup.csv",
            index=False,
        )

    if not corr.empty:
        corr.to_csv(
            output_dir
            / "vp_correlations.csv",
            index=False,
        )

    pairwise.to_csv(
        output_dir
        / "pairwise_strategy_comparisons.csv",
        index=False,
    )

    if not sensitivity.empty:
        sensitivity.to_csv(
            output_dir
            / "lineup_sensitivity.csv",
            index=False,
        )

    print_strategy_table(strategy)

    print()
    print("Seat summary")
    print("=" * 60)
    print(
        seat.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.4f}"
            ),
        )
    )

    print()
    print("Saved analysis to:")
    print(f"  {output_dir}")


if __name__ == "__main__":
    main()
