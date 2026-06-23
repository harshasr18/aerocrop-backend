# ============================================================
# AeroCrop Backend — Flask + Google Earth Engine
# Deploy on Render.com (free tier)
# ============================================================

import os, json, ee, numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ---------- GEE Authentication ----------
def init_gee():
    key_json = os.environ.get('GEE_SERVICE_ACCOUNT_JSON')
    if not key_json:
        raise RuntimeError('GEE_SERVICE_ACCOUNT_JSON env var not set')
    key_data = json.loads(key_json)
    credentials = ee.ServiceAccountCredentials(
        email=key_data['client_email'],
        key_data=json.dumps(key_data)
    )
    ee.Initialize(credentials)

try:
    init_gee()
    print('GEE initialized OK')
except Exception as e:
    print(f'GEE init failed: {e}')

# ---------- Helpers ----------
def mask_s2_clouds(image):
    qa = image.select('QA60')
    cloud_mask = (1 << 10)
    cirrus_mask = (1 << 11)
    mask = (qa.bitwiseAnd(cloud_mask).eq(0)
              .And(qa.bitwiseAnd(cirrus_mask).eq(0)))
    return (image.updateMask(mask)
                 .divide(10000)
                 .copyProperties(image, ['system:time_start']))

def classify_crop(dates, series):
    if len(series) < 4:
        return {'crop': 'Unknown', 'confidence': 0, 'auto': False}

    # Deduplicate same-date readings
    by_date = {}
    for d, v in zip(dates, series):
        by_date.setdefault(d, []).append(v)
    dedup_dates = sorted(by_date.keys())
    dedup_series = [sum(by_date[d])/len(by_date[d]) for d in dedup_dates]

    # 3-point moving average
    smoothed = []
    for i in range(len(dedup_series)):
        lo, hi = max(0, i-1), min(len(dedup_series), i+2)
        smoothed.append(sum(dedup_series[lo:hi]) / (hi - lo))

    peak = max(smoothed)
    peak_idx = smoothed.index(peak)
    from datetime import datetime
    d0 = datetime.strptime(dedup_dates[0], '%Y-%m-%d')
    d_last = datetime.strptime(dedup_dates[-1], '%Y-%m-%d')
    d_peak = datetime.strptime(dedup_dates[peak_idx], '%Y-%m-%d')
    total_span = (d_last - d0).days
    days_since_peak = (d_last - d_peak).days
    end_val = sum(smoothed[-3:]) / min(3, len(smoothed))
    fall_rate = (peak - end_val) / days_since_peak if days_since_peak > 0 else 0

    signatures = {
        'Paddy':     {'peak': 0.75, 'cycleDays': 135, 'fallRate': 0.006},
        'Ragi':      {'peak': 0.55, 'cycleDays': 110, 'fallRate': 0.004},
        'Sugarcane': {'peak': 0.70, 'cycleDays': 330, 'fallRate': 0.001},
        'Maize':     {'peak': 0.65, 'cycleDays': 100, 'fallRate': 0.007},
    }
    best, best_score = None, float('inf')
    for crop, sig in signatures.items():
        score = (abs(peak - sig['peak']) +
                 abs(total_span - sig['cycleDays']) / 100 +
                 abs(fall_rate - sig['fallRate']) * 50)
        if score < best_score:
            best_score, best = score, crop

    confidence = round(max(40, min(95, 100 - best_score * 30)))
    return {'crop': best, 'confidence': confidence, 'auto': True, 'peak': round(peak, 3), 'spanDays': total_span}

def classify_tile(ndvi, smi):
    if ndvi is None or smi is None:
        return 'pending'
    if smi < 0.35:
        level = 'severe'
    elif smi < 0.60:
        level = 'moderate'
    else:
        level = 'healthy'
    if ndvi < 0.30 and level == 'healthy':
        level = 'moderate'
    if ndvi < 0.20 and level == 'moderate':
        level = 'severe'
    return level

def generate_advisory(statuses):
    known = [s for s in statuses if s != 'pending']
    if not known:
        return {'head': 'No data', 'depth': '—', 'duration': '—', 'reason': 'No zones loaded'}
    severe = known.count('severe')
    moderate = known.count('moderate')
    total = len(known)
    if severe / total > 0.15:
        return {'head': 'Irrigate within 24 hours', 'depth': '25–30 mm',
                'duration': '6–8 hrs', 'reason': f'{severe} of {total} zones severely stressed'}
    if moderate / total > 0.25 or severe > 0:
        return {'head': 'Irrigate within 48 hours', 'depth': '15–20 mm',
                'duration': '4–5 hrs', 'reason': f'{moderate} zones moderate, {severe} severe (of {total})'}
    return {'head': 'No irrigation needed', 'depth': '0 mm',
            'duration': '—', 'reason': f'Soil moisture adequate across all {total} zones'}

# ---------- Routes ----------
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'AeroCrop GEE Backend'})

@app.route('/analyse', methods=['POST'])
def analyse():
    try:
        body = request.get_json()
        lat = float(body['lat'])
        lon = float(body['lon'])
        radius = int(body.get('radius', 750))   # metres, default 750
        district = body.get('district', 'Unknown')
        landmark = body.get('landmark', 'Selected location')

        # Build AOI from point + radius
        point = ee.Geometry.Point([lon, lat])
        aoi = point.buffer(radius).bounds()

        # --- Sentinel-2: cloud-masked NDVI composite ---
        s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                .filterBounds(aoi)
                .filterDate('2026-01-01', '2026-06-01')
                .map(mask_s2_clouds))

        ndvi_composite = (s2.median()
                            .normalizedDifference(['B8', 'B4'])
                            .rename('NDVI')
                            .clip(aoi))

        # --- Sentinel-1: despeckled SAR + SMI ---
        s1_baseline = (ee.ImageCollection('COPERNICUS/S1_GRD')
                         .filterBounds(aoi)
                         .filterDate('2024-06-01', '2026-06-01')
                         .filter(ee.Filter.eq('instrumentMode', 'IW'))
                         .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                         .select('VV')
                         .map(lambda img: img.focal_median(30, 'circle', 'meters')
                                             .copyProperties(img, ['system:time_start'])))

        vv_dry = s1_baseline.reduce(ee.Reducer.min()).rename('VV_dry')
        vv_wet = s1_baseline.reduce(ee.Reducer.max()).rename('VV_wet')
        vv_now = (s1_baseline
                    .filterDate(ee.Date(ee.Number(ee.Date(ee.Date.now()).millis()))
                                  .advance(-30, 'day'), ee.Date.now())
                    .median()
                    .rename('VV_current'))

        smi = (vv_now.subtract(vv_dry)
                     .divide(vv_wet.subtract(vv_dry))
                     .clamp(0, 1)
                     .rename('SMI')
                     .clip(aoi))

        combined = ndvi_composite.addBands(smi)

        # --- 8x4 grid sampling ---
        bounds = aoi.bounds().getInfo()['coordinates'][0]
        min_lon = min(p[0] for p in bounds)
        max_lon = max(p[0] for p in bounds)
        min_lat = min(p[1] for p in bounds)
        max_lat = max(p[1] for p in bounds)

        cols, rows = 8, 4
        lon_step = (max_lon - min_lon) / cols
        lat_step = (max_lat - min_lat) / rows

        features = []
        for r in range(rows):
            for c in range(cols):
                lon0 = min_lon + c * lon_step
                lon1 = lon0 + lon_step
                lat0 = max_lat - (r + 1) * lat_step
                lat1 = lat0 + lat_step
                cell = ee.Geometry.Rectangle([lon0, lat0, lon1, lat1])
                features.append(ee.Feature(cell, {'tile_row': r, 'tile_col': c}))

        grid = ee.FeatureCollection(features)
        tile_values = combined.reduceRegions(
            collection=grid, reducer=ee.Reducer.mean(), scale=10)

        # Sort by row*8 + col and extract values
        sorted_tiles = (tile_values
                          .map(lambda f: f.set('idx', ee.Number(f.get('tile_row'))
                                                              .multiply(8)
                                                              .add(f.get('tile_col'))))
                          .sort('idx'))

        ndvi_arr = sorted_tiles.aggregate_array('NDVI').getInfo()
        smi_arr  = sorted_tiles.aggregate_array('SMI').getInfo()

        # --- Crop time series for classifier ---
        season = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                    .filterBounds(aoi)
                    .filterDate('2024-06-01', '2025-02-01')
                    .map(mask_s2_clouds))

        def make_ndvi_feature(img):
            ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
            val = ndvi.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=aoi, scale=10, maxPixels=1e9
            ).get('NDVI')
            return ee.Feature(None, {'date': img.date().format('YYYY-MM-dd'), 'ndvi': val})

        series_fc = (ee.FeatureCollection(season.map(make_ndvi_feature))
                       .filter(ee.Filter.notNull(['ndvi']))
                       .sort('date'))

        crop_dates = series_fc.aggregate_array('date').getInfo()
        crop_ndvi  = series_fc.aggregate_array('ndvi').getInfo()

        # --- Classify and compute advisory ---
        statuses = [classify_tile(n, s) for n, s in zip(ndvi_arr, smi_arr)]
        advisory = generate_advisory(statuses)
        crop_result = classify_crop(crop_dates, crop_ndvi)

        return jsonify({
            'ok': True,
            'district': district,
            'landmark': landmark,
            'lat': lat, 'lon': lon, 'radius': radius,
            'ndvi': ndvi_arr,
            'smi':  smi_arr,
            'statuses': statuses,
            'advisory': advisory,
            'crop': crop_result,
            'bounds': {'minLon': min_lon, 'maxLon': max_lon, 'minLat': min_lat, 'maxLat': max_lat}
        })

    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
