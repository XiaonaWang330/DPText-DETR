"""Analyze P3-DRTP vs baseline visualizations."""
import json
import os

import numpy as np


def main():
    vis_dir = "output/vis_p3"
    summary_path = os.path.join(vis_dir, "summary.json")
    if not os.path.exists(summary_path):
        print(f"Summary not found: {summary_path}")
        return

    with open(summary_path, "r") as f:
        summary = json.load(f)

    # We don't have polygons in summary, so use simple count-based heuristics
    # Find cases where P3 has many more detections than GT (likely FP)
    # and cases where P3 has many fewer than GT (likely FN)

    print("=" * 80)
    print("P3-DRTP Failure Mode Analysis (count-based heuristics)")
    print("=" * 80)

    # Compute proxies
    for s in summary:
        s["p3_fp_proxy"] = max(0, s["p3_num_det"] - s["gt_num"])
        s["p3_fn_proxy"] = max(0, s["gt_num"] - s["p3_num_det"])
        s["bl_fp_proxy"] = max(0, s["bl_num_det"] - s["gt_num"])
        s["bl_fn_proxy"] = max(0, s["gt_num"] - s["bl_num_det"])

    avg_gt = np.mean([s["gt_num"] for s in summary])
    avg_p3 = np.mean([s["p3_num_det"] for s in summary])
    avg_bl = np.mean([s["bl_num_det"] for s in summary])
    avg_p3_fp = np.mean([s["p3_fp_proxy"] for s in summary])
    avg_p3_fn = np.mean([s["p3_fn_proxy"] for s in summary])
    avg_bl_fp = np.mean([s["bl_fp_proxy"] for s in summary])
    avg_bl_fn = np.mean([s["bl_fn_proxy"] for s in summary])

    print(f"\nAverage counts ({len(summary)} images):")
    print(f"  GT:        {avg_gt:.2f}")
    print(f"  Baseline:  {avg_bl:.2f}  (FP proxy={avg_bl_fp:.2f}, FN proxy={avg_bl_fn:.2f})")
    print(f"  P3-DRTP:   {avg_p3:.2f}  (FP proxy={avg_p3_fp:.2f}, FN proxy={avg_p3_fn:.2f})")
    print(f"\nP3 vs Baseline:")
    print(f"  Δ FP proxy: {avg_p3_fp - avg_bl_fp:+.2f} (P3 {'more' if avg_p3_fp > avg_bl_fp else 'fewer'} false positives)")
    print(f"  Δ FN proxy: {avg_p3_fn - avg_bl_fn:+.2f} (P3 {'more' if avg_p3_fn > avg_bl_fn else 'fewer'} missed texts)")

    print("\n" + "=" * 80)
    print("Top 10 cases where P3 likely misses texts (high FN proxy):")
    worst_fn = sorted(summary, key=lambda s: s["p3_fn_proxy"], reverse=True)[:10]
    for s in worst_fn:
        print(f"  {s['image_id']}: P3={s['p3_num_det']}  BL={s['bl_num_det']}  GT={s['gt_num']}  "
              f"P3-FN-proxy={s['p3_fn_proxy']}  BL-FN-proxy={s['bl_fn_proxy']}")

    print("\nTop 10 cases where P3 likely has false positives (high FP proxy):")
    worst_fp = sorted(summary, key=lambda s: s["p3_fp_proxy"], reverse=True)[:10]
    for s in worst_fp:
        print(f"  {s['image_id']}: P3={s['p3_num_det']}  BL={s['bl_num_det']}  GT={s['gt_num']}  "
              f"P3-FP-proxy={s['p3_fp_proxy']}  BL-FP-proxy={s['bl_fp_proxy']}")

    print("\nTop 10 cases where P3 is cleaner than baseline (low FP proxy):")
    best_fp = sorted(summary, key=lambda s: s["p3_fp_proxy"])[:10]
    for s in best_fp:
        print(f"  {s['image_id']}: P3={s['p3_num_det']}  BL={s['bl_num_det']}  GT={s['gt_num']}  "
              f"P3-FP={s['p3_fp_proxy']}  BL-FP={s['bl_fp_proxy']}")

    print("\nTop 10 cases where P3 detects more than baseline (could be recall gain):")
    more_det = sorted(summary, key=lambda s: s["p3_num_det"] - s["bl_num_det"], reverse=True)[:10]
    for s in more_det:
        print(f"  {s['image_id']}: P3={s['p3_num_det']}  BL={s['bl_num_det']}  GT={s['gt_num']}  "
              f"Δ={s['p3_num_det'] - s['bl_num_det']:+d}")


if __name__ == "__main__":
    main()
