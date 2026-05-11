#!/usr/bin/env python3
"""Compare GSM8K results from GPU and Spyre.

Usage:
    python compare_results.py results_gpu.json results_spyre.json
"""

import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_a", help="First results JSON (e.g., GPU)")
    parser.add_argument("file_b", help="Second results JSON (e.g., Spyre)")
    args = parser.parse_args()

    with open(args.file_a) as f:
        data_a = json.load(f)
    with open(args.file_b) as f:
        data_b = json.load(f)

    meta_a = data_a["metadata"]
    meta_b = data_b["metadata"]
    results_a = {r["index"]: r for r in data_a["results"]}
    results_b = {r["index"]: r for r in data_b["results"]}

    label_a = meta_a["backend"]
    label_b = meta_b["backend"]

    print(f"\n{'='*70}")
    print(f"  GSM8K Comparison: {label_a} vs {label_b}")
    print(f"{'='*70}")
    print(f"  Model A: {meta_a['model']} ({label_a})")
    print(f"  Model B: {meta_b['model']} ({label_b})")
    print(f"  Samples: {meta_a['num_samples']} / {meta_b['num_samples']}")
    print()

    common = sorted(set(results_a.keys()) & set(results_b.keys()))

    both_correct = 0
    a_only = 0
    b_only = 0
    both_wrong = 0
    same_pred = 0
    disagreements = []

    for idx in common:
        ra = results_a[idx]
        rb = results_b[idx]
        a_ok = ra["correct"]
        b_ok = rb["correct"]

        if ra["predicted"] == rb["predicted"]:
            same_pred += 1

        if a_ok and b_ok:
            both_correct += 1
        elif a_ok and not b_ok:
            a_only += 1
            disagreements.append((idx, ra, rb))
        elif not a_ok and b_ok:
            b_only += 1
            disagreements.append((idx, ra, rb))
        else:
            both_wrong += 1

    n = len(common)
    acc_a = meta_a["correct"] / meta_a["num_samples"] if meta_a["num_samples"] else 0
    acc_b = meta_b["correct"] / meta_b["num_samples"] if meta_b["num_samples"] else 0

    print(f"  {'Metric':<30} {label_a:<15} {label_b:<15}")
    print(f"  {'-'*60}")
    print(f"  {'Accuracy':<30} {acc_a:<15.1%} {acc_b:<15.1%}")
    print(f"  {'Correct':<30} {meta_a['correct']:<15} {meta_b['correct']:<15}")
    print()
    print(f"  Agreement ({n} common samples):")
    print(f"    Both correct:      {both_correct:>4} ({both_correct/n:.1%})")
    print(f"    {label_a} only:    {a_only:>4} ({a_only/n:.1%})")
    print(f"    {label_b} only:    {b_only:>4} ({b_only/n:.1%})")
    print(f"    Both wrong:        {both_wrong:>4} ({both_wrong/n:.1%})")
    print(f"    Same prediction:   {same_pred:>4} ({same_pred/n:.1%})")

    if disagreements:
        print("\n  Disagreements (first 10):")
        a_pred_label = label_a + " pred"
        b_pred_label = label_b + " pred"
        print(
            f"  {'Idx':<5} {'GT':<10} {a_pred_label:<12} "
            f"{'ok':<4} {b_pred_label:<12} {'ok':<4}"
        )
        print(f"  {'-'*50}")
        for idx, ra, rb in disagreements[:10]:
            print(
                f"  {idx:<5} {ra['ground_truth']:<10} {str(ra['predicted']):<12} "
                f"{'Y' if ra['correct'] else 'N':<4} "
                f"{str(rb['predicted']):<12} {'Y' if rb['correct'] else 'N':<4}"
            )

    # Token-by-token comparison
    has_tokens = "generated_token_ids" in next(
        iter(results_a.values()), {}
    ) and "generated_token_ids" in next(iter(results_b.values()), {})
    if has_tokens:
        print("\n  Token-level comparison:")
        print(
            f"  {'Idx':<5} {'A toks':>7} {'B toks':>7} "
            f"{'Match':>6} {'1st diff':>9} {'Match%':>7}"
        )
        print(f"  {'-'*45}")

        total_first_diff = []
        total_match_pct = []

        for idx in common[:20]:
            toks_a = results_a[idx].get("generated_token_ids", [])
            toks_b = results_b[idx].get("generated_token_ids", [])
            min_len = min(len(toks_a), len(toks_b))

            first_diff = -1
            matches = 0
            for j in range(min_len):
                if toks_a[j] == toks_b[j]:
                    matches += 1
                elif first_diff == -1:
                    first_diff = j

            match_pct = matches / min_len * 100 if min_len > 0 else 0
            exact = first_diff == -1 and len(toks_a) == len(toks_b)
            total_match_pct.append(match_pct)
            if first_diff >= 0:
                total_first_diff.append(first_diff)

            print(
                f"  {idx:<5} {len(toks_a):>7} {len(toks_b):>7} "
                f"{'EXACT' if exact else 'DIFF':>6} "
                f"{'—' if first_diff == -1 else str(first_diff):>9} "
                f"{match_pct:>6.1f}%"
            )

        if total_first_diff:
            avg_fd = sum(total_first_diff) / len(total_first_diff)
            min_fd = min(total_first_diff)
            num_divergent = len(total_first_diff)
            print(
                f"\n  First divergence: avg={avg_fd:.1f}, min={min_fd} "
                f"(of {num_divergent} divergent samples)"
            )
        avg_mp = sum(total_match_pct) / len(total_match_pct) if total_match_pct else 0
        print(f"  Average token match: {avg_mp:.1f}%")

    # Avg generation time
    avg_a = (
        sum(r["time_s"] for r in results_a.values()) / len(results_a)
        if results_a
        else 0
    )
    avg_b = (
        sum(r["time_s"] for r in results_b.values()) / len(results_b)
        if results_b
        else 0
    )
    print(
        f"\n  Avg time/sample:     {avg_a:.1f}s ({label_a})  "
        f"vs  {avg_b:.1f}s ({label_b})"
    )


if __name__ == "__main__":
    main()
