#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd

from services.django_setup import setup_django
from services.step1_problem_qs import build_problem_qs
from services.step2_work_detection import identify_work_done_for_asset_day_inline


@dataclass
class Args:
    day: str
    output: str | None
    django_settings: str | None
    django_path: str | None


def parse_args() -> Args:
    p = argparse.ArgumentParser()
    p.add_argument("--day", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--django-settings", default=None)
    p.add_argument("--django-path", default=None)
    ns = p.parse_args()
    return Args(
        day=str(ns.day),
        output=ns.output,
        django_settings=ns.django_settings,
        django_path=ns.django_path,
    )


def main() -> int:
    args = parse_args()

    setup_django(django_settings=args.django_settings, django_path=args.django_path)

    from ht.apps.real_time_data.models import Asset
    from ht.apps.timescaledb.models import LiveTrackingData

    day = args.day

    start = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = start + timedelta(days=1)

    problem_qs = build_problem_qs(day=day)

    flagged = []
    checked = 0

    for sac in problem_qs.only("id", "tractor_id", "day"):
        checked += 1
        tractor_id = str(sac.tractor_id)

        asset = Asset.objects.filter(tractor_id=tractor_id).order_by("created_at").last()
        if not asset:
            continue

        ltd_rows = list(
            LiveTrackingData.objects.filter(
                asset_id=asset.asset_uuid,
                location_time__gte=start,
                location_time__lt=end,
            )
            .values("location_time", "latitude", "longitude", "speed", "engine_state")
            .order_by("location_time")
        )
        if not ltd_rows:
            continue

        df = pd.DataFrame(ltd_rows)

        diag = identify_work_done_for_asset_day_inline(
            df,
            day=day,
            resample_rule="2min",
            eps_m=20.0,
            min_samples=10,
            min_cluster_duration_min=15.0,
            max_cluster_radius_m=120.0,
            min_cluster_area_ha=0.2,
            min_points_total=40,
            max_linearity_ratio=0.92,
            max_displacement_path_ratio=0.55,
            max_high_speed_ratio=0.35,
            high_speed_kmh=12.0,
            min_engine_on_ratio=0.3,
        )

        if not diag["worked"]:
            continue

        top = diag["clusters"][0] if diag["clusters"] else {}

        print(f"{tractor_id} -> YES (work detected)")

        flagged.append(
            {
                "sac_id": sac.id,
                "tractor_id": tractor_id,
                "day": str(sac.day),
                "asset_uuid": str(asset.asset_uuid),
                "reason_codes": ";".join(diag["reason_codes"]),
                "points_used": diag["points_used"],
                "path_km": diag.get("path_km"),
                "moving_ratio": diag.get("moving_ratio"),
                "top_cluster_duration_min": top.get("duration_min"),
                "top_cluster_area_ha": top.get("hull_area_ha"),
                "top_cluster_radius_m": top.get("radius_m"),
                "top_cluster_linearity": top.get("pca_linearity_ratio"),
                "top_cluster_disp_path_ratio": top.get("disp_path_ratio"),
                "top_cluster_high_speed_ratio": top.get("high_speed_ratio"),
                "top_cluster_engine_on_ratio": top.get("engine_on_ratio"),
                "top_cluster_path_km": top.get("path_km"),
                "top_cluster_net_disp_km": top.get("net_disp_km"),
                "dbscan_eps_m": top.get("eps_m"),
                "dbscan_min_samples": top.get("min_samples"),
            }
        )

    flagged_df = pd.DataFrame(flagged)

    print("Checked:", checked)
    print("Flagged (work detected):", len(flagged_df))

    if len(flagged_df):
        flagged_df = flagged_df.sort_values(
            ["top_cluster_area_ha", "top_cluster_duration_min", "points_used"],
            ascending=False,
        )

    if args.output:
        if args.output.lower().endswith(".csv"):
            flagged_df.to_csv(args.output, index=False)
        else:
            flagged_df.to_parquet(args.output, index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
