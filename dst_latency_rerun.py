

import os, sys, json, time, random, logging
from pathlib import Path
from datetime import datetime

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_ROOT = Path(r"D:\DST_Results")
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_HEADERS = {
    "User-Agent": "DecisionSupportTool/1.0 (contact@nature-demo.eu)",
    "Accept": "application/json",
    "Content-Type": "application/x-www-form-urlencoded",
}
OVERPASS_PACING_SEC = 6
N_REPEATS = 3

INFRA_OPTIONS = {
    "Roads & Highways":        ['["highway"]'],
    "Railways":                ['["railway"]'],
    "Bridges":                 ['["bridge"="yes"]', '["man_made"="bridge"]'],
    "Tunnels":                 ['["tunnel"="yes"]', '["man_made"="tunnel"]'],
    "Dams & Water Storage":    ['["waterway"="dam"]', '["waterway"="weir"]', '["man_made"="dam"]', '["man_made"="dyke"]'],
    "Urban Green Spaces":      ['["leisure"="park"]', '["leisure"="garden"]', '["landuse"="forest"]', '["landuse"="grass"]', '["landuse"="meadow"]', '["natural"="wood"]', '["natural"="wetland"]'],
    "Embankments & Levees":    ['["man_made"="embankment"]', '["man_made"="groyne"]', '["man_made"="levee"]'],
    "Slope Stabilization":     ['["man_made"="check_dam"]', '["man_made"="retaining_wall"]', '["barrier"="retaining_wall"]'],
    "Buildings":               ['["building"]', '["amenity"]'],
    "Power & Utilities":       ['["power"]'],
    "Water Bodies & Rivers":   ['["water"]', '["waterway"]'],
    "Catchment Surface Cover": ['["landuse"]', '["natural"]'],
}


LAND_CENTERS = [
    (52.520, 13.405, "Berlin centre"),
    (48.857, 2.352,  "Paris centre"),
    (51.507, -0.127, "London centre"),
    (40.417, -3.703, "Madrid centre"),
    (41.902, 12.496, "Rome centre"),
    (52.370, 4.895,  "Amsterdam centre"),
    (50.846, 4.353,  "Brussels centre"),
    (47.498, 19.040, "Budapest centre"),
    (50.075, 14.437, "Prague centre"),
    (38.722, -9.139, "Lisbon centre"),
    (53.349, -6.260, "Dublin centre"),
    (59.913, 10.752, "Oslo centre"),
    (45.815, 15.982, "Zagreb centre"),
    (44.426, 26.103, "Bucharest centre"),
    (37.984, 23.728, "Athens centre"),
    (47.270, 11.400, "Innsbruck rural"),
    (45.500, 9.190,  "Milan suburb"),
    (49.610, 6.130,  "Luxembourg rural"),
    (51.946, 4.062,  "Maasvlakte port"),
    (43.296, 5.370,  "Marseille centre"),
    (50.110, 8.682,  "Frankfurt centre"),
    (47.376, 8.541,  "Zurich centre"),
    (54.687, 25.280, "Vilnius centre"),
    (60.169, 24.938, "Helsinki centre"),
    (44.787, 20.448, "Belgrade centre"),
]


RADII_DEG = [0.005, 0.008, 0.012, 0.018, 0.025, 0.035, 0.050]


def setup_logging(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(
        level=logging.INFO, format=fmt,
        handlers=[logging.FileHandler(log_path, mode="w", encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)],
    )


def build_overpass_query(polygon_coords, infra_categories, timeout=90):
    poly_str = " ".join(f"{lat} {lon}" for lat, lon in polygon_coords)
    blocks = []
    for cat in infra_categories:
        for tag_filter in INFRA_OPTIONS[cat]:
            blocks.append(f'  node{tag_filter}(poly:"{poly_str}");')
            blocks.append(f'  way{tag_filter}(poly:"{poly_str}");')
            blocks.append(f'  relation{tag_filter}(poly:"{poly_str}");')
    body = "\n".join(blocks)
    return f"[out:json][timeout:{timeout}];\n(\n{body}\n);\nout center tags;"


def overpass_request(query, max_retries=2):
    for attempt in range(max_retries + 1):
        try:
            r = requests.post(OVERPASS_URL, data={"data": query},
                              headers=OVERPASS_HEADERS, timeout=180)
            if r.status_code == 200: return r
            if r.status_code in (429, 504):
                if attempt < max_retries:
                    logging.warning("Overpass %d, retry in 8 s", r.status_code)
                    time.sleep(8); continue
            logging.error("Overpass HTTP %d", r.status_code); return r
        except requests.exceptions.Timeout:
            if attempt < max_retries:
                logging.warning("timeout, retrying in 8 s"); time.sleep(8); continue
            return None
        except requests.exceptions.RequestException as e:
            logging.error("network error: %s", e); return None
    return None


def deg_to_km2(lat, radius_deg):
    """Approximate area of a 2*radius square at given latitude, in km²."""
    R = 6371.0
    deg_to_km_lat = (np.pi / 180) * R
    deg_to_km_lon = (np.pi / 180) * R * np.cos(np.radians(lat))
    return (2 * radius_deg * deg_to_km_lat) * (2 * radius_deg * deg_to_km_lon)


def main():
    out_dir = OUTPUT_ROOT / "latency_polygons"
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir / "latency_run.log")
    logging.info("Latency re-run started at %s", datetime.now().isoformat())

    random.seed(7)
    polygons = []
    radii_cycle = (RADII_DEG * 4)[:len(LAND_CENTERS)]
    random.shuffle(radii_cycle)
    for i, ((cy, cx, name), r) in enumerate(zip(LAND_CENTERS, radii_cycle)):
        if len(polygons) >= 20: break
        polygons.append({"id": f"P{i:02d}", "name": name,
                         "center": (cy, cx), "radius_deg": r})

    all_infras = list(INFRA_OPTIONS.keys())
    rows = []

    for idx, p in enumerate(polygons):
        cy, cx = p["center"]; r = p["radius_deg"]
        coords = [(cy - r, cx - r), (cy - r, cx + r),
                  (cy + r, cx + r), (cy + r, cx - r)]
        area = deg_to_km2(cy, r)
        query = build_overpass_query(coords, all_infras)

        latencies = []
        n_elements = None
        for rep in range(N_REPEATS):
            logging.info("[%s] %d/%d  rep %d/%d  area=%.2f km²  %s",
                         p["id"], idx + 1, len(polygons), rep + 1, N_REPEATS,
                         area, p["name"])
            t0 = time.perf_counter()
            r_resp = overpass_request(query)
            dt = time.perf_counter() - t0
            if r_resp is not None and r_resp.status_code == 200:
                latencies.append(dt)
                ne = len(r_resp.json().get("elements", []))
                if n_elements is None: n_elements = ne
                logging.info("    rep %d: %.2fs  (n_elements=%d)", rep + 1, dt, ne)
            else:
                logging.warning("    rep %d: FAILED", rep + 1)
            time.sleep(OVERPASS_PACING_SEC)

        if not latencies or n_elements is None:
            logging.warning("[%s] all reps failed; skipping", p["id"])
            continue

        row = {
            "polygon_id": p["id"], "name": p["name"],
            "center_lat": cy, "center_lon": cx,
            "area_km2": area, "n_elements": n_elements,
            "latency_mean_s": float(np.mean(latencies)),
            "latency_sd_s": float(np.std(latencies, ddof=1)) if len(latencies) > 1 else 0.0,
            "latency_min_s": float(min(latencies)),
            "latency_max_s": float(max(latencies)),
            "n_reps": len(latencies),
        }
        rows.append(row)
        (out_dir / f"run_{p['id']}.json").write_text(json.dumps(row, indent=2))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "_latency_summary_v2.csv", index=False)
    logging.info("Wrote summary with %d polygons", len(df))

    if not df.empty:
        make_figure6(df, OUTPUT_ROOT / "figure6_latency_vs_elements.png")
        make_figure6_supplementary(df, OUTPUT_ROOT / "figure6_supplement.png")

    logging.info("DONE at %s", datetime.now().isoformat())


def make_figure6(df, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))

    ax = axes[0]
    df_pos = df[df["n_elements"] > 0].copy()
    if not df_pos.empty:
        ax.errorbar(df_pos["n_elements"], df_pos["latency_mean_s"],
                    yerr=df_pos["latency_sd_s"], fmt="o", color="#2D6A4F",
                    ecolor="#888", capsize=3, alpha=0.85,
                    label=f"n={len(df_pos)} polygons (mean ± SD over 3 reps)")

        x = df_pos["n_elements"].values
        y = df_pos["latency_mean_s"].values
        if (x > 0).sum() >= 2:
            a = float(np.sum(x * y) / np.sum(x * x))
            xs = np.linspace(max(1, x.min()), x.max(), 100)
            ax.plot(xs, a * xs, "--", color="#666",
                    label=f"linear fit: y = {a*1000:.2f} ms × n")
        ax.legend(fontsize=9, loc="upper left")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("returned OSM elements"); ax.set_ylabel("Overpass latency (s)")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title("(a) Latency vs. element count", fontsize=10)

    ax = axes[1]
    ax.errorbar(df["area_km2"], df["latency_mean_s"],
                yerr=df["latency_sd_s"], fmt="s", color="#1B4332",
                ecolor="#888", capsize=3, alpha=0.85)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("polygon area (km²)"); ax.set_ylabel("Overpass latency (s)")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title("(b) Latency vs. polygon area", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved %s", out_png)


def make_figure6_supplementary(df, out_png):
    """Per-polygon variability: shows that server-load noise is a real factor."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    df_sorted = df.sort_values("n_elements")
    x = np.arange(len(df_sorted))
    ax.bar(x, df_sorted["latency_mean_s"], yerr=df_sorted["latency_sd_s"],
           color="#52B788", edgecolor="#1B4332", capsize=3)
    labels = [f"{row.polygon_id}\n(n={int(row.n_elements)})" for row in df_sorted.itertuples()]
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Overpass latency (s)")
    ax.set_title("Per-polygon latency (mean ± SD over 3 repetitions), sorted by element count",
                 fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved %s", out_png)


if __name__ == "__main__":
    main()
