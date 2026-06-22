import os, json, datetime
import ee
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AeroCrop Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── GEE initialisation via service account ────────────────────────────────────
def init_gee():
    key_json = os.environ.get('GEE_SERVICE_ACCOUNT_JSON')
    if not key_json:
        raise EnvironmentError(
            "GEE_SERVICE_ACCOUNT_JSON environment variable not set. "
            "Add it in Render → Environment → Add environment variable."
        )
    key_dict = json.loads(key_json)
    credentials = ee.ServiceAccountCredentials(
        key_dict['client_email'],
        key_data=key_json
    )
    ee.Initialize(credentials)
    print(f"GEE initialized as {key_dict['client_email']}")

init_gee()

# ── Sentinel-2 cloud mask ─────────────────────────────────────────────────────
def mask_s2_clouds(image):
    qa = image.select('QA60')
    mask = (qa.bitwiseAnd(1 << 10).eq(0)
              .And(qa.bitwiseAnd(1 << 11).eq(0)))
    return (image.updateMask(mask)
                 .divide(10000)
                 .copyProperties(image, ['system:time_start']))

# ── Advisory logic (mirrors the frontend classifier) ─────────────────────────
def compute_advisory(ndvi_arr, smi_arr):
    def classify(ndvi, smi):
        if ndvi is None or smi is None:
            return 'pending'
        level = 'severe' if smi < 0.35 else ('moderate' if smi < 0.60 else 'healthy')
        if ndvi < 0.30 and level == 'healthy':
            level = 'moderate'
        if ndvi < 0.20 and level == 'moderate':
            level = 'severe'
        return level

    statuses = [classify(n, s) for n, s in zip(ndvi_arr, smi_arr)]
    known = [s for s in statuses if s != 'pending']
    total = len(known)
    if total == 0:
        return {'head': 'Awaiting data', 'depth': '—', 'duration': '—', 'reason': 'No valid tiles'}

    severe = known.count('severe')
    moderate = known.count('moderate')

    if severe / total > 0.15:
        return {'head': 'Irrigate within 24 hours', 'depth': '25–30 mm', 'duration': '6–8 hrs',
                'reason': f'{severe} of {total} zones severely stressed'}
    if moderate / total > 0.25 or severe > 0:
        return {'head': 'Irrigate within 48 hours', 'depth': '15–20 mm', 'duration': '4–5 hrs',
                'reason': f'{moderate} zones moderate, {severe} severe (of {total})'}
    return {'head': 'No irrigation needed', 'depth': '0 mm', 'duration': '—',
            'reason': f'Soil moisture adequate across all {total} zones'}

# ── Main analysis endpoint ────────────────────────────────────────────────────
@app.get("/analyze")
def analyze(lat: float, lon: float, radius_m: int = 750):
    """
    Live GEE analysis for any lat/lon in Karnataka.
    Returns 32-tile NDVI + SMI values + computed irrigation advisory.
    Typical response time: 15–25 seconds.
    """
    try:
        today        = datetime.date.today()
        one_year_ago = today.replace(year=today.year - 1)
        thirty_ago   = today - datetime.timedelta(days=30)
        ninety_ago   = today - datetime.timedelta(days=90)

        center = ee.Geometry.Point([lon, lat])
        aoi    = center.buffer(radius_m).bounds()

        # ── Sentinel-2 NDVI ──────────────────────────────────────────────────
        s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                .filterBounds(aoi)
                .filterDate(one_year_ago.isoformat(), today.isoformat())
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
                .map(mask_s2_clouds))

        # Fallback: widen cloud threshold if not enough images
        if s2.size().getInfo() < 3:
            s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                    .filterBounds(aoi)
                    .filterDate(one_year_ago.isoformat(), today.isoformat())
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60))
                    .map(mask_s2_clouds))

        ndvi_img = (s2.median()
                      .normalizedDifference(['B8', 'B4'])
                      .rename('NDVI')
                      .clip(aoi))

        # ── Sentinel-1 SAR soil moisture index ───────────────────────────────
        s1_all = (ee.ImageCollection('COPERNICUS/S1_GRD')
                    .filterBounds(aoi)
                    .filterDate(one_year_ago.isoformat(), today.isoformat())
                    .filter(ee.Filter.eq('instrumentMode', 'IW'))
                    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                    .select('VV')
                    .map(lambda img: img.focal_median(30, 'circle', 'meters')
                                       .copyProperties(img, ['system:time_start'])))

        vv_dry = s1_all.reduce(ee.Reducer.min()).rename('VV_dry')
        vv_wet = s1_all.reduce(ee.Reducer.max()).rename('VV_wet')

        # Current: last 30 days, fallback to 90 days
        vv_recent = s1_all.filterDate(thirty_ago.isoformat(), today.isoformat())
        if vv_recent.size().getInfo() == 0:
            vv_recent = s1_all.filterDate(ninety_ago.isoformat(), today.isoformat())

        vv_current = vv_recent.median().rename('VV_current')

        smi = (vv_current.subtract(vv_dry)
                         .divide(vv_wet.subtract(vv_dry))
                         .clamp(0, 1)
                         .rename('SMI')
                         .clip(aoi))

        combined = ndvi_img.addBands(smi)

        # ── 8×4 tile grid centred on AOI ─────────────────────────────────────
        bbox   = aoi.bounds().getInfo()['coordinates'][0]
        lons   = [p[0] for p in bbox]
        lats   = [p[1] for p in bbox]
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)

        cols, rows = 8, 4
        lon_step = (max_lon - min_lon) / cols
        lat_step = (max_lat - min_lat) / rows

        grid = []
        for r in range(rows):
            for c in range(cols):
                lo = min_lon + c * lon_step
                la0 = max_lat - (r + 1) * lat_step
                grid.append(ee.Feature(
                    ee.Geometry.Rectangle([lo, la0, lo + lon_step, la0 + lat_step]),
                    {'tile_row': r, 'tile_col': c}
                ))

        tile_values = combined.reduceRegions(
            collection=ee.FeatureCollection(grid),
            reducer=ee.Reducer.mean(),
            scale=10
        )

        features = sorted(
            tile_values.getInfo()['features'],
            key=lambda f: f['properties']['tile_row'] * 8 + f['properties']['tile_col']
        )

        ndvi_arr = [f['properties'].get('NDVI') for f in features]
        smi_arr  = [f['properties'].get('SMI')  for f in features]
        advisory = compute_advisory(ndvi_arr, smi_arr)

        return {
            "status":   "success",
            "lat":       lat,
            "lon":       lon,
            "radius_m":  radius_m,
            "ndvi":      ndvi_arr,
            "smi":       smi_arr,
            "advisory":  advisory
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/health")
def health():
    return {"status": "ok", "message": "AeroCrop backend is running"}
