import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


def _to_utc_datetime(s):
    return pd.to_datetime(s, utc=True, errors="coerce")


def _meters_projection(df, lat_col="latitude", lon_col="longitude"):
    lat = df[lat_col].to_numpy(dtype=float)
    lon = df[lon_col].to_numpy(dtype=float)
    lat0 = np.nanmedian(lat)
    meters_per_deg_lat = 110540.0
    meters_per_deg_lon = 111320.0 * np.cos(np.radians(lat0))
    x = lon * meters_per_deg_lon
    y = lat * meters_per_deg_lat
    return np.column_stack([x, y])


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return r * c


def _poly_area_m2(points_xy):
    if len(points_xy) < 3:
        return 0.0
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _compute_path_length_km(df):
    if len(df) < 2:
        return 0.0
    lat = df["latitude"].to_numpy(dtype=float)
    lon = df["longitude"].to_numpy(dtype=float)
    lat2 = np.roll(lat, -1)
    lon2 = np.roll(lon, -1)
    dist = _haversine_km(lat[:-1], lon[:-1], lat2[:-1], lon2[:-1])
    dist = np.nan_to_num(dist, nan=0.0, posinf=0.0, neginf=0.0)
    return float(dist.sum())


def _compute_net_displacement_km(df):
    if len(df) < 2:
        return 0.0
    lat1 = float(df["latitude"].iloc[0])
    lon1 = float(df["longitude"].iloc[0])
    lat2 = float(df["latitude"].iloc[-1])
    lon2 = float(df["longitude"].iloc[-1])
    return float(_haversine_km(lat1, lon1, lat2, lon2))


def _compute_moving_ratio(df):
    if "speed" not in df.columns:
        return None
    spd = pd.to_numeric(df["speed"], errors="coerce").fillna(0)
    return float((spd > 0.5).mean())


def _compute_high_speed_ratio(df, high_kmh=12.0):
    if "speed" not in df.columns:
        return None
    spd = pd.to_numeric(df["speed"], errors="coerce").fillna(0)
    return float((spd > float(high_kmh)).mean())


def _compute_engine_on_ratio(df):
    if "engine_state" not in df.columns:
        return None
    es = df["engine_state"].astype(str).str.upper()
    return float((es == "ON").mean())


def _split_sessions(df, max_gap_min=20.0):
    if max_gap_min is None:
        return [df]
    gaps = df["location_time"].diff().dt.total_seconds().div(60.0)
    session_id = (gaps.fillna(0) > float(max_gap_min)).cumsum()
    return [g.reset_index(drop=True) for _, g in df.groupby(session_id, sort=True)]


def _safe_hull_area_ha(sub_xy):
    if sub_xy is None or len(sub_xy) < 3:
        return 0.0

    sub_xy_u = np.unique(np.round(sub_xy, 2), axis=0)
    if len(sub_xy_u) < 3:
        return 0.0

    if float(np.std(sub_xy_u[:, 0])) < 0.5 or float(np.std(sub_xy_u[:, 1])) < 0.5:
        return 0.0

    try:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(sub_xy_u, qhull_options="QJ")
        hull_xy = sub_xy_u[hull.vertices]
        return _poly_area_m2(hull_xy) / 10000.0
    except Exception:
        return 0.0


def _pca_linearity_ratio(xy):
    if xy is None or len(xy) < 3:
        return 1.0
    x = xy - np.mean(xy, axis=0, keepdims=True)
    cov = np.cov(x.T)
    if not np.isfinite(cov).all():
        return 1.0
    vals = np.linalg.eigvalsh(cov)
    vals = np.sort(vals)[::-1]
    denom = float(vals.sum())
    if denom <= 0:
        return 1.0
    return float(vals[0] / denom)


def identify_work_done_for_asset_day_inline(
    df_raw,
    *,
    day=None,
    resample_rule="2min",
    max_session_gap_min=20.0,
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
):
    df = df_raw.copy()
    df["location_time"] = _to_utc_datetime(df["location_time"])
    df = df.dropna(subset=["location_time", "latitude", "longitude"]).sort_values("location_time")

    if day is not None:
        day_dt = pd.to_datetime(day, utc=True, errors="coerce")
        if pd.isna(day_dt):
            return {"worked": False, "reason_codes": ["bad_day"], "clusters": []}
        start = day_dt.normalize()
        end = start + pd.Timedelta(days=1)
        df = df[(df["location_time"] >= start) & (df["location_time"] < end)]

    points_total = int(len(df))
    if points_total == 0:
        return {
            "worked": False,
            "reason_codes": ["no_points"],
            "points_total": 0,
            "points_used": 0,
            "path_km": 0.0,
            "moving_ratio": None,
            "clusters": [],
        }

    if resample_rule is not None:
        df = (
            df.set_index("location_time")
            .resample(resample_rule)
            .last()
            .reset_index()
            .dropna(subset=["location_time", "latitude", "longitude"])
            .sort_values("location_time")
            .reset_index(drop=True)
        )

    points_used = int(len(df))
    moving_ratio = _compute_moving_ratio(df)
    path_km = _compute_path_length_km(df)

    reason_codes = []
    if points_used < int(min_points_total):
        reason_codes.append("too_few_points")
    if moving_ratio is not None and moving_ratio < 0.02:
        reason_codes.append("low_moving_ratio")

    eps_list = [float(eps_m), float(eps_m) * 1.5, float(eps_m) * 2.5]
    eps_list = [e for e in eps_list if e > 0]
    eps_list = [e if e <= 150 else 150.0 for e in eps_list]

    sessions = _split_sessions(df, max_gap_min=max_session_gap_min)

    clusters_out = []

    for sess in sessions:
        if len(sess) < max(3, int(min_samples), int(min_points_total)):
            continue

        xy = _meters_projection(sess)

        ms_list = [int(min_samples), max(5, int(min_samples) // 2), 3]
        ms_list = [ms for ms in ms_list if ms <= len(sess)]

        for ms_try in ms_list:
            found_any = False
            for eps_try in eps_list:
                labels = DBSCAN(eps=float(eps_try), min_samples=int(ms_try)).fit_predict(xy)

                for cl in sorted(set(int(x) for x in labels)):
                    if cl == -1:
                        continue

                    idx = np.where(labels == cl)[0]
                    sub = sess.iloc[idx]
                    if len(sub) < int(min_points_total):
                        continue

                    start_t = sub["location_time"].min()
                    end_t = sub["location_time"].max()
                    duration_min = float((end_t - start_t).total_seconds() / 60.0)
                    if duration_min < float(min_cluster_duration_min):
                        continue

                    centroid_lat = float(sub["latitude"].mean())
                    centroid_lon = float(sub["longitude"].mean())

                    sub_xy = _meters_projection(sub)
                    cxy = _meters_projection(
                        pd.DataFrame({"latitude": [centroid_lat], "longitude": [centroid_lon]})
                    )[0]
                    radius_m = float(np.sqrt(((sub_xy - cxy) ** 2).sum(axis=1)).max())
                    if radius_m > float(max_cluster_radius_m):
                        continue

                    hull_area_ha = _safe_hull_area_ha(sub_xy)
                    if hull_area_ha < float(min_cluster_area_ha):
                        continue

                    pca_ratio = _pca_linearity_ratio(sub_xy)
                    if pca_ratio > float(max_linearity_ratio):
                        continue

                    path_len = _compute_path_length_km(sub)
                    net_disp = _compute_net_displacement_km(sub)
                    disp_path_ratio = (net_disp / path_len) if path_len > 0 else 1.0
                    if disp_path_ratio > float(max_displacement_path_ratio):
                        continue

                    hs_ratio = _compute_high_speed_ratio(sub, high_kmh=high_speed_kmh)
                    if hs_ratio is not None and hs_ratio > float(max_high_speed_ratio):
                        continue

                    eng_on_ratio = _compute_engine_on_ratio(sub)
                    if eng_on_ratio is not None and eng_on_ratio < float(min_engine_on_ratio):
                        continue

                    clusters_out.append(
                        dict(
                            points=int(len(sub)),
                            start_time=start_t,
                            end_time=end_t,
                            duration_min=duration_min,
                            centroid_lat=centroid_lat,
                            centroid_lon=centroid_lon,
                            radius_m=radius_m,
                            hull_area_ha=float(hull_area_ha),
                            pca_linearity_ratio=float(pca_ratio),
                            path_km=float(path_len),
                            net_disp_km=float(net_disp),
                            disp_path_ratio=float(disp_path_ratio),
                            high_speed_ratio=hs_ratio,
                            engine_on_ratio=eng_on_ratio,
                            eps_m=float(eps_try),
                            min_samples=int(ms_try),
                        )
                    )
                    found_any = True

                if found_any:
                    break
            if found_any:
                break

    clusters_out.sort(key=lambda c: (c["duration_min"], c["points"], c["hull_area_ha"]), reverse=True)

    worked = bool(clusters_out)
    reason_codes.append("dbscan_cluster_found" if worked else "no_dbscan_cluster")

    return {
        "worked": bool(worked),
        "reason_codes": reason_codes,
        "points_total": points_total,
        "points_used": points_used,
        "path_km": float(path_km),
        "moving_ratio": moving_ratio,
        "clusters": clusters_out,
    }
