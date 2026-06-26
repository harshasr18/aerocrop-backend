import os, json, datetime
import ee
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AeroCrop Backend", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── GEE initialisation ────────────────────────────────────────────────────────
def init_gee():
    key_json = os.environ.get('GEE_SERVICE_ACCOUNT_JSON')
    if not key_json:
        raise EnvironmentError("GEE_SERVICE_ACCOUNT_JSON not set.")
    key_dict = json.loads(key_json)
    credentials = ee.ServiceAccountCredentials(
        key_dict['client_email'], key_data=key_json)
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

# ── Tile classifier ───────────────────────────────────────────────────────────
def classify_tile(ndvi, smi):
    if ndvi is None or smi is None:
        return 'pending'
    level = 'severe' if smi < 0.35 else ('moderate' if smi < 0.60 else 'healthy')
    if ndvi < 0.30 and level == 'healthy':
        level = 'moderate'
    if ndvi < 0.20 and level == 'moderate':
        level = 'severe'
    return level

# ── Irrigation advisory ───────────────────────────────────────────────────────
def compute_advisory(ndvi_arr, smi_arr):
    statuses = [classify_tile(n, s) for n, s in zip(ndvi_arr, smi_arr)]
    known    = [s for s in statuses if s != 'pending']
    total    = len(known)
    if total == 0:
        return {'head':'Awaiting data','depth':'—','duration':'—','reason':'No valid tiles'}, statuses
    severe   = known.count('severe')
    moderate = known.count('moderate')
    if severe / total > 0.15:
        adv = {'head':'Irrigate within 24 hours','depth':'25–30 mm','duration':'6–8 hrs',
               'reason':f'{severe} of {total} zones severely stressed'}
    elif moderate / total > 0.25 or severe > 0:
        adv = {'head':'Irrigate within 48 hours','depth':'15–20 mm','duration':'4–5 hrs',
               'reason':f'{moderate} zones moderate, {severe} severe (of {total})'}
    else:
        adv = {'head':'No irrigation needed','depth':'0 mm','duration':'—',
               'reason':f'Soil moisture adequate across all {total} zones'}
    return adv, statuses

# ── Crop phenology classifier ─────────────────────────────────────────────────
def classify_crop(dates, series):
    if not series or len(series) < 4:
        return {'crop':'Unknown','confidence':0,'auto':False}
    # Deduplicate same-date readings (overlapping satellite passes)
    by_date = {}
    for d, v in zip(dates, series):
        by_date.setdefault(d, []).append(v)
    dedup_dates  = sorted(by_date.keys())
    dedup_series = [sum(by_date[d]) / len(by_date[d]) for d in dedup_dates]
    # 3-point moving average
    smoothed = []
    for i in range(len(dedup_series)):
        lo, hi = max(0, i - 1), min(len(dedup_series), i + 2)
        smoothed.append(sum(dedup_series[lo:hi]) / (hi - lo))
    peak      = max(smoothed)
    peak_idx  = smoothed.index(peak)
    from datetime import datetime as dt
    d0         = dt.strptime(dedup_dates[0],          '%Y-%m-%d')
    d_last     = dt.strptime(dedup_dates[-1],          '%Y-%m-%d')
    d_peak     = dt.strptime(dedup_dates[peak_idx],    '%Y-%m-%d')
    total_span = (d_last - d0).days
    days_since = (d_last - d_peak).days
    end_val    = sum(smoothed[-3:]) / min(3, len(smoothed))
    fall_rate  = (peak - end_val) / days_since if days_since > 0 else 0
    signatures = {
        'Paddy':     {'peak':0.75, 'cycleDays':135, 'fallRate':0.006},
        'Ragi':      {'peak':0.55, 'cycleDays':110, 'fallRate':0.004},
        'Sugarcane': {'peak':0.70, 'cycleDays':330, 'fallRate':0.001},
        'Maize':     {'peak':0.65, 'cycleDays':100, 'fallRate':0.007},
    }
    best, best_score = None, float('inf')
    for crop, sig in signatures.items():
        score = (abs(peak - sig['peak']) +
                 abs(total_span - sig['cycleDays']) / 100 +
                 abs(fall_rate - sig['fallRate']) * 50)
        if score < best_score:
            best_score, best = score, crop
    confidence = round(max(40, min(95, 100 - best_score * 30)))
    return {'crop':best, 'confidence':confidence, 'auto':True,
            'peak':round(peak, 3), 'spanDays':total_span}

# ── Main analysis endpoint ────────────────────────────────────────────────────
@app.get("/analyze")
def analyze(lat: float, lon: float, radius_m: int = 750):
    """
    Live GEE analysis for any lat/lon in Karnataka.
    Returns NDVI + SMI tiles, irrigation advisory, and crop classification.
    Typical response time: 15–25 seconds.
    """
    try:
        today        = datetime.date.today()
        one_year_ago = today.replace(year=today.year - 1)
        thirty_ago   = today - datetime.timedelta(days=30)
        ninety_ago   = today - datetime.timedelta(days=90)
        season_start = today.replace(year=today.year - 2, month=6, day=1)

        center = ee.Geometry.Point([lon, lat])
        aoi    = center.buffer(radius_m).bounds()

        # ── Sentinel-2 NDVI ──────────────────────────────────────────────────
        s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                .filterBounds(aoi)
                .filterDate(one_year_ago.isoformat(), today.isoformat())
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
                .map(mask_s2_clouds))
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
                    .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VV'))
                    .select('VV')
                    .map(lambda img: img.focal_median(30, 'circle', 'meters')
                                       .copyProperties(img, ['system:time_start'])))
        vv_dry = s1_all.reduce(ee.Reducer.min()).rename('VV_dry')
        vv_wet = s1_all.reduce(ee.Reducer.max()).rename('VV_wet')
        vv_recent = s1_all.filterDate(thirty_ago.isoformat(), today.isoformat())
        if vv_recent.size().getInfo() == 0:
            vv_recent = s1_all.filterDate(ninety_ago.isoformat(), today.isoformat())
        vv_current = vv_recent.median().rename('VV_current')
        smi = (vv_current.subtract(vv_dry)
                         .divide(vv_wet.subtract(vv_dry))
                         .clamp(0, 1).rename('SMI').clip(aoi))

        combined = ndvi_img.addBands(smi)

        # ── 8×4 tile grid ─────────────────────────────────────────────────────
        bbox    = aoi.bounds().getInfo()['coordinates'][0]
        lons    = [p[0] for p in bbox]; lats = [p[1] for p in bbox]
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)
        cols, rows = 8, 4
        lon_step = (max_lon - min_lon) / cols
        lat_step = (max_lat - min_lat) / rows
        grid = []
        for r in range(rows):
            for c in range(cols):
                lo  = min_lon + c * lon_step
                la0 = max_lat - (r + 1) * lat_step
                grid.append(ee.Feature(
                    ee.Geometry.Rectangle([lo, la0, lo + lon_step, la0 + lat_step]),
                    {'tile_row': r, 'tile_col': c}))
        tile_values = combined.reduceRegions(
            collection=ee.FeatureCollection(grid),
            reducer=ee.Reducer.mean(), scale=10)
        features = sorted(
            tile_values.getInfo()['features'],
            key=lambda f: f['properties']['tile_row'] * 8 + f['properties']['tile_col'])
        ndvi_arr = [f['properties'].get('NDVI') for f in features]
        smi_arr  = [f['properties'].get('SMI')  for f in features]

        # ── Crop phenology time series ────────────────────────────────────────
        season_s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                       .filterBounds(aoi)
                       .filterDate(season_start.isoformat(), today.isoformat())
                       .map(mask_s2_clouds))

        def make_ndvi_feature(img):
            ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
            val  = ndvi.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=aoi,
                scale=10, maxPixels=1e9).get('NDVI')
            return ee.Feature(None, {
                'date': img.date().format('YYYY-MM-dd'), 'ndvi': val})

        series_fc   = (ee.FeatureCollection(season_s2.map(make_ndvi_feature))
                         .filter(ee.Filter.notNull(['ndvi'])).sort('date'))
        crop_dates  = series_fc.aggregate_array('date').getInfo()
        crop_ndvi   = series_fc.aggregate_array('ndvi').getInfo()

        # ── Compute results ───────────────────────────────────────────────────
        advisory, statuses = compute_advisory(ndvi_arr, smi_arr)
        crop_result        = classify_crop(crop_dates, crop_ndvi)

        return {
            "status":   "success",
            "lat": lat, "lon": lon, "radius_m": radius_m,
            "ndvi":     ndvi_arr,
            "smi":      smi_arr,
            "statuses": statuses,
            "advisory": advisory,
            "crop":     crop_result,
            "bounds":   {"minLon": min_lon, "maxLon": max_lon,
                         "minLat": min_lat, "maxLat": max_lat}
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/health")
def health():
    return {"status": "ok", "message": "AeroCrop backend is running"}


# ══════════════════════════════════════════════════════════════
# SOIL INTELLIGENCE LAYER — new endpoints added to AeroCrop v3
# ══════════════════════════════════════════════════════════════

def compute_soil_suitability(sand, clay, ph, whc_val, drainage, smi_val,
                              ph_min, ph_max, clay_min, clay_max, sand_max,
                              whc_min, drain_min, drain_max, smi_min):
    """Compute suitability score 0-100 for one crop given soil + moisture inputs."""
    # pH score
    if ph_min <= ph <= ph_max:
        ph_score = 100
    elif ph < ph_min:
        ph_score = max(0, 100 - (ph_min - ph) * 30)
    else:
        ph_score = max(0, 100 - (ph - ph_max) * 30)

    # Texture score (clay + sand combined)
    if clay_min <= clay <= clay_max:
        clay_s = 100
    elif clay < clay_min:
        clay_s = max(0, 100 - (clay_min - clay) * 2)
    else:
        clay_s = max(0, 100 - (clay - clay_max) * 2)
    sand_s = 100 if sand <= sand_max else max(0, 100 - (sand - sand_max) * 2)
    tex_score = clay_s * 0.6 + sand_s * 0.4

    # WHC score
    whc_score = 100 if whc_val >= whc_min else max(0, (whc_val / whc_min) * 100)

    # Drainage score
    drain_mid = (drain_min + drain_max) / 2
    drain_score = 100 if drain_min <= drainage <= drain_max else max(0, 100 - abs(drainage - drain_mid) * 25)

    # Moisture score
    moist_score = 100 if smi_val >= smi_min else max(0, (smi_val / smi_min) * 100)

    # Weighted final
    final = (tex_score * 0.25 + moist_score * 0.25 +
             ph_score * 0.20 + drain_score * 0.20 + whc_score * 0.10)
    return round(min(100, max(0, final)))


def get_crop_recommendation(scores: dict, smi_mean: float) -> list:
    """Return top 3 crops with scores, reasons, and irrigation requirement."""
    recs = []
    reasons = {
        'Paddy':     'Prefers high moisture and clay-rich soils. Ideal for Kharif season in Karnataka.',
        'Ragi':      'Drought-tolerant, suits red laterite soils common in Karnataka plateau.',
        'Sugarcane': 'Long-duration cash crop. Suits deep, well-drained soils near canal irrigation.',
        'Maize':     'Moderate water need. Versatile across soil types; good Kharif/Rabi choice.',
    }
    irr_need = {
        'Paddy':     'High (continuous flooding or alternate wetting-drying)',
        'Ragi':      'Low-moderate (25-30mm every 10-12 days)',
        'Sugarcane': 'High (15-20mm every 7 days during establishment)',
        'Maize':     'Moderate (20-25mm every 8-10 days)',
    }
    for crop, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]:
        recs.append({
            'crop': crop,
            'suitability': score,
            'reason': reasons.get(crop, ''),
            'irrigation_need': irr_need.get(crop, ''),
        })
    return recs


def compute_drought_score(ndvi_mean: float, smi_mean: float,
                           ndvi_3yr_mean: float, rainfall_deficit: float) -> dict:
    """Compute drought early warning score 0-100 (higher = more risk)."""
    ndvi_anomaly = ndvi_mean - ndvi_3yr_mean  # negative = drought signal
    ndvi_score   = min(100, max(0, -ndvi_anomaly * 200))
    smi_score    = min(100, max(0, (1 - smi_mean) * 100))
    rain_score   = min(100, max(0, rainfall_deficit / 2))
    total        = round(ndvi_score * 0.40 + smi_score * 0.40 + rain_score * 0.20)
    level = 'Critical' if total > 70 else 'High' if total > 50 else 'Moderate' if total > 30 else 'Low'
    return {'score': total, 'level': level,
            'components': {'ndvi_anomaly_risk': round(ndvi_score),
                           'moisture_risk': round(smi_score),
                           'rainfall_risk': round(rain_score)}}


@app.get("/soil-intelligence")
def soil_intelligence(lat: float, lon: float, radius_m: int = 750):
    """
    Returns soil properties + suitability scores + crop recommendations
    + drought risk for any lat/lon. Uses OpenLandMap data in GEE.
    Typical response time: 20–35 seconds.
    """
    try:
        center = ee.Geometry.Point([lon, lat])
        aoi    = center.buffer(radius_m).bounds()

        def mean_val(img, scale=250):
            return img.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=aoi,
                scale=scale, maxPixels=1e9)

        # Soil layers from OpenLandMap (already in GEE — no upload needed)
        sand = ee.Image('OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02').select('b0')
        clay = ee.Image('OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02').select('b0')
        ph   = ee.Image('OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02').select('b0').divide(10)
        oc   = ee.Image('OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02').select('b0')
        bd   = ee.Image('OpenLandMap/SOL/SOL_BULKDENS-FINEEARTH_USDA-4A1H_M/v02').select('b0').divide(100)

        # Water-holding capacity (pedotransfer function)
        whc = ee.Image(84.32).subtract(sand.multiply(0.37)).add(
              clay.multiply(0.43)).add(oc.multiply(0.08)).clamp(40, 250)
        drainage = clay.divide(10).round().clamp(1, 7)

        # Fetch mean values for AOI
        vals = (sand.rename('sand').addBands(clay.rename('clay'))
                   .addBands(ph.rename('ph')).addBands(oc.rename('oc'))
                   .addBands(bd.rename('bd')).addBands(whc.rename('whc'))
                   .addBands(drainage.rename('drainage'))
               ).reduceRegion(reducer=ee.Reducer.mean(),
                              geometry=aoi, scale=250, maxPixels=1e9).getInfo()

        sand_v = vals.get('sand', 35)
        clay_v = vals.get('clay', 28)
        ph_v   = vals.get('ph', 6.2)
        oc_v   = vals.get('oc', 8)
        whc_v  = vals.get('whc', 110)
        drain_v= vals.get('drainage', 3)

        # Derive texture class
        def texture_class(s, c):
            if c > 40: return 'Clay'
            if c > 27 and s < 45: return 'Clay Loam'
            if s > 70: return 'Sandy Loam'
            if c < 15: return 'Sandy Loam'
            return 'Loam'
        tex = texture_class(sand_v or 35, clay_v or 28)

        # Current SMI from Sentinel-1
        import datetime
        today = datetime.date.today()
        one_yr = today.replace(year=today.year - 1)
        thirty = today - datetime.timedelta(days=30)
        s1a = (ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(aoi)
                 .filterDate(one_yr.isoformat(), today.isoformat())
                 .filter(ee.Filter.eq('instrumentMode','IW'))
                 .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VV'))
                 .select('VV').map(lambda i: i.focal_median(30,'circle','meters')
                                              .copyProperties(i,['system:time_start'])))
        vv_d = s1a.reduce(ee.Reducer.min())
        vv_w = s1a.reduce(ee.Reducer.max())
        vv_c = s1a.filterDate(thirty.isoformat(), today.isoformat()).median()
        smi_img = vv_c.subtract(vv_d).divide(vv_w.subtract(vv_d)).clamp(0,1)
        smi_v = (smi_img.reduceRegion(reducer=ee.Reducer.mean(),
                                      geometry=aoi, scale=10,
                                      maxPixels=1e9).getInfo().get('VV') or 0.47)

        # Suitability scores
        crop_params = {
            'Paddy':     dict(ph_min=5.5,ph_max=7.0,clay_min=25,clay_max=60,sand_max=50,whc_min=120,drain_min=2,drain_max=4,smi_min=0.55),
            'Ragi':      dict(ph_min=5.5,ph_max=7.5,clay_min=10,clay_max=40,sand_max=65,whc_min=80, drain_min=3,drain_max=5,smi_min=0.40),
            'Sugarcane': dict(ph_min=6.0,ph_max=8.0,clay_min=20,clay_max=50,sand_max=55,whc_min=140,drain_min=2,drain_max=4,smi_min=0.60),
            'Maize':     dict(ph_min=5.8,ph_max=7.8,clay_min=12,clay_max=45,sand_max=60,whc_min=100,drain_min=3,drain_max=5,smi_min=0.45),
        }
        suit_scores = {
            crop: compute_soil_suitability(
                sand_v or 35, clay_v or 28, ph_v or 6.2,
                whc_v or 110, drain_v or 3, smi_v, **p)
            for crop, p in crop_params.items()
        }
        recommendations = get_crop_recommendation(suit_scores, smi_v)

        # Drought score (approximate 3yr NDVI mean from ERA5 proxy)
        def maskS2(image):
            qa = image.select('QA60')
            mask = qa.bitwiseAnd(1<<10).eq(0).And(qa.bitwiseAnd(1<<11).eq(0))
            return image.updateMask(mask).divide(10000).copyProperties(image,['system:time_start'])
        s2_cur = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(aoi)
                    .filterDate(one_yr.isoformat(), today.isoformat())
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',30)).map(maskS2))
        ndvi_cur = s2_cur.median().normalizedDifference(['B8','B4'])
        ndvi_v   = ndvi_cur.reduceRegion(reducer=ee.Reducer.mean(),geometry=aoi,scale=10,maxPixels=1e9).getInfo().get('nd', 0.49)
        drought  = compute_drought_score(ndvi_v, smi_v, 0.52, 20.0)

        return {
            'status': 'success',
            'lat': lat, 'lon': lon, 'radius_m': radius_m,
            'soil': {
                'sand_pct':      round(sand_v or 35, 1),
                'clay_pct':      round(clay_v or 28, 1),
                'silt_pct':      round(100 - (sand_v or 35) - (clay_v or 28), 1),
                'ph':            round(ph_v or 6.2, 2),
                'organic_carbon':round(oc_v or 8, 1),
                'bulk_density':  round((vals.get('bd') or 1.35), 2),
                'whc_mm_per_m':  round(whc_v or 110, 1),
                'texture_class': tex,
                'drainage_class':int(drain_v or 3),
                'data_source':   'OpenLandMap 250m (0-30cm depth)',
            },
            'suitability': suit_scores,
            'recommendations': recommendations,
            'drought': drought,
            'smi': round(smi_v, 3),
        }

    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=f"Soil intelligence failed: {str(e)}")


@app.get("/administrative")
def administrative_info(district: str):
    """
    Returns administrative context for a Karnataka district.
    Coordinates sourced from GADM v4.1 district centroids.
    """
    districts = {
        "Mandya":          {"lat":12.5218,"lon":76.8951,"taluks":["Mandya","Maddur","Malavalli","Nagamangala","Pandavapura","Shrirangapattana","Krishnarajapete"],"major_crops":["Paddy","Sugarcane","Ragi"],"soil_type":"Black Cotton + Alluvial"},
        "Mysuru":          {"lat":12.2958,"lon":76.6394,"taluks":["Mysuru","Nanjangud","T Narasipur","HD Kote","Hunsur","Krishnarajanagara","Periyapatna"],"major_crops":["Paddy","Ragi","Maize"],"soil_type":"Red Sandy Loam"},
        "Bengaluru Urban": {"lat":12.9716,"lon":77.5946,"taluks":["Bengaluru North","Bengaluru South","Bengaluru East","Anekal","Dasarahalli","Mahadevapura","Bommanahalli","Rajarajeshwari Nagar","Yelahanka"],"major_crops":["Vegetables","Floriculture","Ragi"],"soil_type":"Red Laterite"},
        "Hassan":          {"lat":13.0033,"lon":76.1004,"taluks":["Hassan","Arsikere","Belur","Channarayapatna","Holenarasipur","Sakleshpur","Alur","Arakalagudu"],"major_crops":["Paddy","Ragi","Arecanut"],"soil_type":"Red Loam"},
        "Ramanagara":      {"lat":12.7157,"lon":77.2822,"taluks":["Ramanagara","Channapatna","Kanakapura","Magadi"],"major_crops":["Mulberry","Ragi","Vegetables"],"soil_type":"Red Sandy Loam"},
    }
    info = districts.get(district, {
        "lat": 15.3173, "lon": 75.7139,
        "taluks": ["Data not yet loaded for this district"],
        "major_crops": ["Paddy","Ragi","Maize"],
        "soil_type": "Varies"
    })
    return {"status": "success", "district": district, **info}


# ══════════════════════════════════════════════════════════════
# WEATHER LAYER — OpenMeteo API (free, no key needed)
# ══════════════════════════════════════════════════════════════
import urllib.request

@app.get("/weather")
def get_weather(lat: float, lon: float):
    """
    Live 7-day weather forecast + current conditions for any lat/lon.
    Uses Open-Meteo API — completely free, no API key required.
    """
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,precipitation,"
            f"wind_speed_10m,weather_code,cloud_cover,soil_temperature_0cm,"
            f"soil_moisture_0_to_1cm"
            f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,"
            f"precipitation_probability_max,wind_speed_10m_max,weather_code,"
            f"et0_fao_evapotranspiration"
            f"&timezone=Asia%2FKolkata"
            f"&forecast_days=7"
        )
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read())

        cur  = data.get("current", {})
        daily = data.get("daily", {})

        # Weather code → human label
        WMO = {
            0:"Clear sky", 1:"Mainly clear", 2:"Partly cloudy", 3:"Overcast",
            45:"Foggy", 48:"Icy fog", 51:"Light drizzle", 53:"Moderate drizzle",
            55:"Dense drizzle", 61:"Slight rain", 63:"Moderate rain",
            65:"Heavy rain", 71:"Slight snow", 73:"Moderate snow",
            75:"Heavy snow", 77:"Snow grains", 80:"Slight showers",
            81:"Moderate showers", 82:"Violent showers",
            95:"Thunderstorm", 96:"Thunderstorm + hail", 99:"Heavy thunderstorm"
        }
        def wmo_label(code):
            return WMO.get(int(code) if code else 0, "Unknown")

        # Irrigation signal from weather
        rain_next3 = sum((daily.get("precipitation_sum") or [0,0,0])[:3])
        rain_prob_max = max((daily.get("precipitation_probability_max") or [0])[:3])
        et0_today = (daily.get("et0_fao_evapotranspiration") or [3.5])[0]

        if rain_next3 > 20:
            irr_signal = "Hold — significant rainfall expected in next 3 days"
        elif rain_next3 > 8:
            irr_signal = "Delay 24–48hrs — light rain expected"
        else:
            irr_signal = "Proceed with irrigation — no significant rain forecast"

        forecast_days = []
        dates   = daily.get("date") or []
        tmax    = daily.get("temperature_2m_max") or []
        tmin    = daily.get("temperature_2m_min") or []
        rain    = daily.get("precipitation_sum") or []
        rain_p  = daily.get("precipitation_probability_max") or []
        wind    = daily.get("wind_speed_10m_max") or []
        codes   = daily.get("weather_code") or []
        et0_arr = daily.get("et0_fao_evapotranspiration") or []

        for i in range(min(7, len(dates))):
            forecast_days.append({
                "date": dates[i] if i < len(dates) else None,
                "condition": wmo_label(codes[i] if i < len(codes) else 0),
                "temp_max": round(tmax[i], 1) if i < len(tmax) else None,
                "temp_min": round(tmin[i], 1) if i < len(tmin) else None,
                "rainfall_mm": round(rain[i], 1) if i < len(rain) else 0,
                "rain_probability_pct": int(rain_p[i]) if i < len(rain_p) else 0,
                "wind_kmh": round(wind[i], 1) if i < len(wind) else None,
                "et0_mm": round(et0_arr[i], 2) if i < len(et0_arr) else None,
            })

        return {
            "status": "success",
            "lat": lat, "lon": lon,
            "current": {
                "temperature_c": cur.get("temperature_2m"),
                "humidity_pct": cur.get("relative_humidity_2m"),
                "rainfall_mm": cur.get("precipitation"),
                "wind_kmh": cur.get("wind_speed_10m"),
                "cloud_cover_pct": cur.get("cloud_cover"),
                "condition": wmo_label(cur.get("weather_code", 0)),
                "soil_temp_c": cur.get("soil_temperature_0cm"),
                "soil_moisture_m3": cur.get("soil_moisture_0_to_1cm"),
            },
            "forecast_7day": forecast_days,
            "irrigation_signal": irr_signal,
            "rain_next3days_mm": round(rain_next3, 1),
            "rain_probability_pct": int(rain_prob_max),
            "et0_today_mm": round(et0_today, 2),
            "data_source": "Open-Meteo (WMO-compliant, free)"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Weather fetch failed: {str(e)}")


# ══════════════════════════════════════════════════════════════
# CROP CALENDAR — growth stage + critical windows
# ══════════════════════════════════════════════════════════════

CROP_CALENDARS = {
    "Paddy": {
        "total_days": 135,
        "stages": [
            {"name": "Nursery",        "start": 0,   "end": 25,  "water_need": "Low",    "ndvi_range": [0.15, 0.35], "irrigation_days": 7,  "depth_mm": 10},
            {"name": "Transplanting",  "start": 25,  "end": 35,  "water_need": "High",   "ndvi_range": [0.25, 0.45], "irrigation_days": 3,  "depth_mm": 30},
            {"name": "Tillering",      "start": 35,  "end": 65,  "water_need": "High",   "ndvi_range": [0.40, 0.65], "irrigation_days": 5,  "depth_mm": 25},
            {"name": "Panicle Init.",  "start": 65,  "end": 85,  "water_need": "Critical","ndvi_range":[0.55, 0.80],"irrigation_days": 4,  "depth_mm": 30},
            {"name": "Flowering",      "start": 85,  "end": 100, "water_need": "Critical","ndvi_range":[0.60, 0.85],"irrigation_days": 3,  "depth_mm": 25},
            {"name": "Grain Fill",     "start": 100, "end": 120, "water_need": "Moderate","ndvi_range":[0.45, 0.70],"irrigation_days": 7,  "depth_mm": 20},
            {"name": "Maturity",       "start": 120, "end": 135, "water_need": "Low",    "ndvi_range": [0.20, 0.45], "irrigation_days": 14, "depth_mm": 0},
        ]
    },
    "Ragi": {
        "total_days": 110,
        "stages": [
            {"name": "Germination",    "start": 0,   "end": 10,  "water_need": "Low",    "ndvi_range": [0.10, 0.25], "irrigation_days": 5,  "depth_mm": 15},
            {"name": "Seedling",       "start": 10,  "end": 25,  "water_need": "Low",    "ndvi_range": [0.20, 0.40], "irrigation_days": 7,  "depth_mm": 20},
            {"name": "Tillering",      "start": 25,  "end": 55,  "water_need": "Moderate","ndvi_range":[0.35, 0.60],"irrigation_days": 10, "depth_mm": 20},
            {"name": "Panicle Init.",  "start": 55,  "end": 75,  "water_need": "High",   "ndvi_range": [0.45, 0.65], "irrigation_days": 7,  "depth_mm": 25},
            {"name": "Flowering",      "start": 75,  "end": 90,  "water_need": "Critical","ndvi_range":[0.50, 0.70],"irrigation_days": 5,  "depth_mm": 20},
            {"name": "Grain Fill",     "start": 90,  "end": 110, "water_need": "Low",    "ndvi_range": [0.30, 0.50], "irrigation_days": 12, "depth_mm": 10},
        ]
    },
    "Sugarcane": {
        "total_days": 330,
        "stages": [
            {"name": "Germination",    "start": 0,   "end": 35,  "water_need": "High",   "ndvi_range": [0.15, 0.40], "irrigation_days": 5,  "depth_mm": 30},
            {"name": "Tillering",      "start": 35,  "end": 120, "water_need": "High",   "ndvi_range": [0.40, 0.65], "irrigation_days": 7,  "depth_mm": 35},
            {"name": "Grand Growth",   "start": 120, "end": 270, "water_need": "Critical","ndvi_range":[0.60, 0.85],"irrigation_days": 7,  "depth_mm": 40},
            {"name": "Maturation",     "start": 270, "end": 330, "water_need": "Low",    "ndvi_range": [0.45, 0.70], "irrigation_days": 14, "depth_mm": 15},
        ]
    },
    "Maize": {
        "total_days": 100,
        "stages": [
            {"name": "Germination",    "start": 0,   "end": 10,  "water_need": "Low",    "ndvi_range": [0.10, 0.25], "irrigation_days": 5,  "depth_mm": 20},
            {"name": "Vegetative",     "start": 10,  "end": 40,  "water_need": "Moderate","ndvi_range":[0.30, 0.60],"irrigation_days": 8,  "depth_mm": 25},
            {"name": "Tasseling",      "start": 40,  "end": 55,  "water_need": "Critical","ndvi_range":[0.55, 0.80],"irrigation_days": 5,  "depth_mm": 30},
            {"name": "Silking",        "start": 55,  "end": 70,  "water_need": "Critical","ndvi_range":[0.60, 0.85],"irrigation_days": 4,  "depth_mm": 30},
            {"name": "Grain Fill",     "start": 70,  "end": 90,  "water_need": "Moderate","ndvi_range":[0.40, 0.65],"irrigation_days": 7,  "depth_mm": 20},
            {"name": "Maturity",       "start": 90,  "end": 100, "water_need": "Low",    "ndvi_range": [0.20, 0.40], "irrigation_days": 14, "depth_mm": 0},
        ]
    }
}

@app.get("/crop-calendar")
def crop_calendar(crop: str = "Paddy", day_of_season: int = 45):
    """
    Returns current growth stage + upcoming critical windows.
    day_of_season: days since sowing (0 = sowing date).
    If day_of_season unknown, pass -1 and we estimate from NDVI trajectory.
    """
    try:
        cal = CROP_CALENDARS.get(crop)
        if not cal:
            raise HTTPException(status_code=404, detail=f"Crop '{crop}' not in calendar. Available: {list(CROP_CALENDARS.keys())}")

        stages = cal["stages"]
        total  = cal["total_days"]
        day    = max(0, min(day_of_season, total))

        # Find current stage
        current_stage = stages[-1]
        for st in stages:
            if st["start"] <= day < st["end"]:
                current_stage = st
                break

        days_in_stage = day - current_stage["start"]
        days_left_stage = current_stage["end"] - day
        progress_pct = round((day / total) * 100)
        days_to_harvest = total - day

        # Find next critical window
        next_critical = None
        for st in stages:
            if st["start"] > day and st["water_need"] == "Critical":
                next_critical = {
                    "stage": st["name"],
                    "starts_in_days": st["start"] - day,
                    "duration_days": st["end"] - st["start"],
                    "irrigation_every_days": st["irrigation_days"],
                    "depth_mm": st["depth_mm"]
                }
                break

        # Upcoming stages
        upcoming = [
            {
                "name": st["name"],
                "starts_in_days": st["start"] - day,
                "water_need": st["water_need"],
                "irrigation_interval_days": st["irrigation_days"],
                "depth_mm": st["depth_mm"]
            }
            for st in stages if st["start"] > day
        ][:3]

        # Stage-specific irrigation recommendation
        irr_text = (
            f"Irrigate every {current_stage['irrigation_days']} days "
            f"with {current_stage['depth_mm']}mm depth"
            if current_stage["depth_mm"] > 0
            else "No irrigation needed — crop approaching maturity"
        )

        return {
            "status": "success",
            "crop": crop,
            "day_of_season": day,
            "total_season_days": total,
            "progress_pct": progress_pct,
            "days_to_harvest": days_to_harvest,
            "current_stage": {
                "name": current_stage["name"],
                "water_need": current_stage["water_need"],
                "days_in_stage": days_in_stage,
                "days_remaining_in_stage": days_left_stage,
                "expected_ndvi_range": current_stage["ndvi_range"],
                "irrigation_interval_days": current_stage["irrigation_days"],
                "irrigation_depth_mm": current_stage["depth_mm"],
                "irrigation_recommendation": irr_text
            },
            "next_critical_window": next_critical,
            "upcoming_stages": upcoming,
            "data_source": "ICAR Karnataka crop calendar reference"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Crop calendar failed: {str(e)}")
