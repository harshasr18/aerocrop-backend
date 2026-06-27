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

# ══════════════════════════════════════════════════════════════
# AEROCROP — COMPLETE 31-CROP MODULE
# Covers all major Karnataka crops with:
# - Phenological NDVI signatures (for crop classification)
# - Soil suitability parameters (ICAR Karnataka reference)
# - Growth stage calendars (stage-wise water need)
# - Irrigation requirements
# - Agronomic reasons for recommendation
# ══════════════════════════════════════════════════════════════

CROP_SIGNATURES = {
    "Rice (Paddy)": {"peak":0.75,"cycleDays":135,"fallRate":0.006},
    "Ragi": {"peak":0.55,"cycleDays":110,"fallRate":0.004},
    "Maize": {"peak":0.65,"cycleDays":100,"fallRate":0.007},
    "Jowar": {"peak":0.6,"cycleDays":115,"fallRate":0.005},
    "Bajra": {"peak":0.55,"cycleDays":85,"fallRate":0.006},
    "Toor Dal": {"peak":0.55,"cycleDays":165,"fallRate":0.003},
    "Chickpea": {"peak":0.5,"cycleDays":100,"fallRate":0.005},
    "Green Gram": {"peak":0.5,"cycleDays":65,"fallRate":0.007},
    "Black Gram": {"peak":0.48,"cycleDays":75,"fallRate":0.006},
    "Groundnut": {"peak":0.58,"cycleDays":125,"fallRate":0.005},
    "Sunflower": {"peak":0.6,"cycleDays":95,"fallRate":0.007},
    "Soybean": {"peak":0.65,"cycleDays":100,"fallRate":0.006},
    "Sugarcane": {"peak":0.7,"cycleDays":330,"fallRate":0.001},
    "Cotton": {"peak":0.6,"cycleDays":185,"fallRate":0.003},
    "Coffee": {"peak":0.65,"cycleDays":365,"fallRate":0.0005},
    "Coconut": {"peak":0.7,"cycleDays":365,"fallRate":0.0003},
    "Arecanut": {"peak":0.72,"cycleDays":365,"fallRate":0.0003},
    "Mango": {"peak":0.68,"cycleDays":365,"fallRate":0.0004},
    "Banana": {"peak":0.65,"cycleDays":300,"fallRate":0.001},
    "Grapes": {"peak":0.58,"cycleDays":180,"fallRate":0.004},
    "Pomegranate": {"peak":0.55,"cycleDays":180,"fallRate":0.003},
    "Tomato": {"peak":0.55,"cycleDays":90,"fallRate":0.008},
    "Onion": {"peak":0.45,"cycleDays":110,"fallRate":0.006},
    "Potato": {"peak":0.6,"cycleDays":100,"fallRate":0.008},
    "Chilli": {"peak":0.52,"cycleDays":150,"fallRate":0.004},
    "Brinjal": {"peak":0.52,"cycleDays":120,"fallRate":0.005},
    "Cabbage": {"peak":0.5,"cycleDays":90,"fallRate":0.008},
    "Cauliflower": {"peak":0.48,"cycleDays":90,"fallRate":0.008},
    "Turmeric": {"peak":0.62,"cycleDays":270,"fallRate":0.002},
    "Ginger": {"peak":0.6,"cycleDays":210,"fallRate":0.002},
    "Black Pepper": {"peak":0.68,"cycleDays":365,"fallRate":0.0004},
}

CROP_SOIL_PARAMS = {
    "Rice (Paddy)": dict(ph_min=5.5,ph_max=7.0,clay_min=25,clay_max=60,sand_max=50,whc_min=120,drain_min=2,drain_max=4,smi_min=0.55),
    "Ragi": dict(ph_min=5.5,ph_max=7.5,clay_min=10,clay_max=40,sand_max=65,whc_min=80,drain_min=3,drain_max=5,smi_min=0.4),
    "Maize": dict(ph_min=5.8,ph_max=7.8,clay_min=12,clay_max=45,sand_max=60,whc_min=100,drain_min=3,drain_max=5,smi_min=0.45),
    "Jowar": dict(ph_min=6.0,ph_max=8.0,clay_min=10,clay_max=40,sand_max=65,whc_min=75,drain_min=3,drain_max=6,smi_min=0.35),
    "Bajra": dict(ph_min=6.0,ph_max=8.0,clay_min=8,clay_max=35,sand_max=70,whc_min=60,drain_min=3,drain_max=6,smi_min=0.3),
    "Toor Dal": dict(ph_min=6.0,ph_max=7.5,clay_min=10,clay_max=35,sand_max=60,whc_min=80,drain_min=3,drain_max=6,smi_min=0.35),
    "Chickpea": dict(ph_min=6.0,ph_max=8.0,clay_min=10,clay_max=35,sand_max=60,whc_min=70,drain_min=3,drain_max=6,smi_min=0.3),
    "Green Gram": dict(ph_min=6.0,ph_max=7.5,clay_min=10,clay_max=35,sand_max=65,whc_min=70,drain_min=3,drain_max=5,smi_min=0.35),
    "Black Gram": dict(ph_min=5.5,ph_max=7.5,clay_min=10,clay_max=35,sand_max=65,whc_min=70,drain_min=3,drain_max=5,smi_min=0.35),
    "Groundnut": dict(ph_min=6.0,ph_max=7.0,clay_min=8,clay_max=28,sand_max=70,whc_min=80,drain_min=3,drain_max=5,smi_min=0.4),
    "Sunflower": dict(ph_min=6.0,ph_max=7.5,clay_min=10,clay_max=40,sand_max=60,whc_min=90,drain_min=3,drain_max=5,smi_min=0.4),
    "Soybean": dict(ph_min=6.0,ph_max=7.5,clay_min=12,clay_max=40,sand_max=55,whc_min=100,drain_min=3,drain_max=5,smi_min=0.45),
    "Sugarcane": dict(ph_min=6.0,ph_max=8.0,clay_min=20,clay_max=50,sand_max=55,whc_min=140,drain_min=2,drain_max=4,smi_min=0.6),
    "Cotton": dict(ph_min=6.0,ph_max=8.0,clay_min=25,clay_max=55,sand_max=50,whc_min=110,drain_min=2,drain_max=5,smi_min=0.45),
    "Coffee": dict(ph_min=5.5,ph_max=6.5,clay_min=20,clay_max=45,sand_max=55,whc_min=130,drain_min=2,drain_max=4,smi_min=0.55),
    "Coconut": dict(ph_min=5.5,ph_max=7.5,clay_min=15,clay_max=40,sand_max=60,whc_min=120,drain_min=2,drain_max=5,smi_min=0.5),
    "Arecanut": dict(ph_min=6.0,ph_max=7.5,clay_min=20,clay_max=45,sand_max=55,whc_min=130,drain_min=2,drain_max=4,smi_min=0.55),
    "Mango": dict(ph_min=5.5,ph_max=7.5,clay_min=15,clay_max=40,sand_max=60,whc_min=110,drain_min=3,drain_max=5,smi_min=0.4),
    "Banana": dict(ph_min=6.0,ph_max=7.5,clay_min=20,clay_max=45,sand_max=55,whc_min=130,drain_min=2,drain_max=4,smi_min=0.6),
    "Grapes": dict(ph_min=6.0,ph_max=7.5,clay_min=15,clay_max=40,sand_max=60,whc_min=100,drain_min=3,drain_max=5,smi_min=0.42),
    "Pomegranate": dict(ph_min=5.5,ph_max=7.5,clay_min=12,clay_max=38,sand_max=62,whc_min=90,drain_min=3,drain_max=6,smi_min=0.38),
    "Tomato": dict(ph_min=6.0,ph_max=7.0,clay_min=15,clay_max=40,sand_max=60,whc_min=90,drain_min=3,drain_max=5,smi_min=0.45),
    "Onion": dict(ph_min=6.0,ph_max=7.5,clay_min=12,clay_max=38,sand_max=62,whc_min=85,drain_min=3,drain_max=5,smi_min=0.4),
    "Potato": dict(ph_min=5.0,ph_max=6.5,clay_min=12,clay_max=35,sand_max=62,whc_min=100,drain_min=3,drain_max=5,smi_min=0.48),
    "Chilli": dict(ph_min=6.0,ph_max=7.5,clay_min=12,clay_max=38,sand_max=60,whc_min=90,drain_min=3,drain_max=5,smi_min=0.42),
    "Brinjal": dict(ph_min=5.5,ph_max=7.5,clay_min=12,clay_max=38,sand_max=62,whc_min=85,drain_min=3,drain_max=5,smi_min=0.42),
    "Cabbage": dict(ph_min=6.0,ph_max=7.5,clay_min=15,clay_max=40,sand_max=58,whc_min=90,drain_min=3,drain_max=5,smi_min=0.45),
    "Cauliflower": dict(ph_min=6.0,ph_max=7.5,clay_min=15,clay_max=40,sand_max=58,whc_min=90,drain_min=3,drain_max=5,smi_min=0.45),
    "Turmeric": dict(ph_min=5.5,ph_max=7.5,clay_min=18,clay_max=45,sand_max=58,whc_min=120,drain_min=2,drain_max=4,smi_min=0.52),
    "Ginger": dict(ph_min=5.5,ph_max=6.5,clay_min=15,clay_max=42,sand_max=60,whc_min=115,drain_min=2,drain_max=4,smi_min=0.5),
    "Black Pepper": dict(ph_min=5.5,ph_max=7.0,clay_min=20,clay_max=45,sand_max=55,whc_min=130,drain_min=2,drain_max=4,smi_min=0.55),
}

CROP_IRRIGATION_NEED = {
    "Rice (Paddy)": "High (regular irrigation every 4-7 days)",
    "Ragi": "Low (minimal irrigation — rain-fed suitable in most zones)",
    "Maize": "Moderate (irrigation every 7-10 days depending on stage)",
    "Jowar": "Low (minimal irrigation — rain-fed suitable in most zones)",
    "Bajra": "Very Low (drought-tolerant — irrigate only at critical stages)",
    "Toor Dal": "Low (minimal irrigation — rain-fed suitable in most zones)",
    "Chickpea": "Very Low (drought-tolerant — irrigate only at critical stages)",
    "Green Gram": "Low (minimal irrigation — rain-fed suitable in most zones)",
    "Black Gram": "Low (minimal irrigation — rain-fed suitable in most zones)",
    "Groundnut": "Moderate (irrigation every 7-10 days depending on stage)",
    "Sunflower": "Moderate (irrigation every 7-10 days depending on stage)",
    "Soybean": "Moderate (irrigation every 7-10 days depending on stage)",
    "Sugarcane": "Very High (continuous flooding or alternate wetting-drying)",
    "Cotton": "Moderate (irrigation every 7-10 days depending on stage)",
    "Coffee": "High (regular irrigation every 4-7 days)",
    "Coconut": "High (regular irrigation every 4-7 days)",
    "Arecanut": "High (regular irrigation every 4-7 days)",
    "Mango": "Moderate (irrigation every 7-10 days depending on stage)",
    "Banana": "Very High (continuous flooding or alternate wetting-drying)",
    "Grapes": "Moderate (irrigation every 7-10 days depending on stage)",
    "Pomegranate": "Moderate (irrigation every 7-10 days depending on stage)",
    "Tomato": "High (regular irrigation every 4-7 days)",
    "Onion": "Moderate (irrigation every 7-10 days depending on stage)",
    "Potato": "High (regular irrigation every 4-7 days)",
    "Chilli": "Moderate (irrigation every 7-10 days depending on stage)",
    "Brinjal": "Moderate (irrigation every 7-10 days depending on stage)",
    "Cabbage": "High (regular irrigation every 4-7 days)",
    "Cauliflower": "High (regular irrigation every 4-7 days)",
    "Turmeric": "High (regular irrigation every 4-7 days)",
    "Ginger": "High (regular irrigation every 4-7 days)",
    "Black Pepper": "High (regular irrigation every 4-7 days)",
}

CROP_REASONS = {
    "Rice (Paddy)": "Dominant Kharif cereal in Karnataka. Suits waterlogged clay soils and high rainfall zones.",
    "Ragi": "Drought-tolerant staple. Ideal for red laterite soils of Karnataka plateau. Excellent food security crop.",
    "Maize": "High-yield cereal suitable for both Kharif and Rabi. Versatile across Karnataka's agro-climatic zones.",
    "Jowar": "Drought-resistant cereal for northern and dry zone Karnataka. Low water requirement.",
    "Bajra": "Extremely drought-tolerant. Suited for light sandy soils of north Karnataka.",
    "Toor Dal": "Major pulse crop of Karnataka. Nitrogen-fixing. Ideal for mixed cropping systems.",
    "Chickpea": "Rabi pulse suited for black cotton soils. Minimal irrigation requirement.",
    "Green Gram": "Short-duration pulse. Excellent for crop rotation and soil health.",
    "Black Gram": "High-protein pulse. Suited for loamy to clayey soils of Karnataka.",
    "Groundnut": "Major oilseed of Karnataka. Suited for light sandy loam soils. High economic value.",
    "Sunflower": "High-oil oilseed. Suited for both Kharif and Rabi. Tolerates moderate drought.",
    "Soybean": "Protein-rich oilseed. Kharif crop suited for black cotton soils of northern Karnataka.",
    "Sugarcane": "High-value commercial crop. Dominates Mandya, Belgaum command areas. Needs irrigation canal access.",
    "Cotton": "Major commercial crop of northern Karnataka. Suited for black cotton (Vertisol) soils.",
    "Coffee": "Premium plantation crop of Kodagu and Hassan. Requires shaded, well-drained slopes.",
    "Coconut": "Important plantation crop of coastal and transitional Karnataka zones.",
    "Arecanut": "High-value plantation crop of Shimoga, Chikkamagaluru, Hassan districts.",
    "Mango": "Karnataka's leading fruit crop. Suited across diverse soil types from Kolar to Kodagu.",
    "Banana": "High-value fruit with consistent demand. Suited for deep, well-irrigated soils.",
    "Grapes": "Premium fruit for Vijayapura, Belagavi districts. Requires well-drained soils and trellis.",
    "Pomegranate": "Drought-tolerant fruit crop. Suited for semi-arid zones of Karnataka.",
    "Tomato": "High-value vegetable. Leading crop in Kolar and Chikkaballapur districts.",
    "Onion": "Major vegetable crop of Gadag, Haveri, Belgaum. High market demand.",
    "Potato": "Cool-season vegetable. Suited for Rabi season in Hassan, Chikkamagaluru highland areas.",
    "Chilli": "Important spice-vegetable crop. Byadgi chilli is a GI-tagged Karnataka variety.",
    "Brinjal": "Widely grown vegetable across Karnataka. Adapted to diverse soil and climate conditions.",
    "Cabbage": "Cool-season vegetable. High demand. Suited for Rabi season across Karnataka.",
    "Cauliflower": "Premium cool-season vegetable. Suited for Rabi cropping in Karnataka highlands.",
    "Turmeric": "High-value spice with medicinal use. Suited for humid zones of south Karnataka.",
    "Ginger": "Premium spice crop. Grown extensively in Hassan, Kodagu, Chikkamagaluru districts.",
    "Black Pepper": "King of spices. Grown as a perennial vine in Karnataka's western ghats.",
}

CROP_CALENDARS = {
    "Rice (Paddy)": {
        "total_days": 135,
        "category": "Cereal",
        "season": "Kharif",
        "water": "High",
        "stages": [
            {"name":"Nursery","start":0,"end":25,"water_need":"Low","ndvi_range":[0.15, 0.35],"irrigation_days":7,"depth_mm":10},
            {"name":"Transplanting","start":25,"end":35,"water_need":"High","ndvi_range":[0.25, 0.45],"irrigation_days":3,"depth_mm":30},
            {"name":"Tillering","start":35,"end":65,"water_need":"High","ndvi_range":[0.4, 0.65],"irrigation_days":5,"depth_mm":25},
            {"name":"Panicle Initiation","start":65,"end":85,"water_need":"Critical","ndvi_range":[0.55, 0.8],"irrigation_days":4,"depth_mm":30},
            {"name":"Flowering","start":85,"end":100,"water_need":"Critical","ndvi_range":[0.6, 0.85],"irrigation_days":3,"depth_mm":25},
            {"name":"Grain Fill","start":100,"end":120,"water_need":"Moderate","ndvi_range":[0.45, 0.7],"irrigation_days":7,"depth_mm":20},
            {"name":"Maturity","start":120,"end":135,"water_need":"Low","ndvi_range":[0.2, 0.45],"irrigation_days":14,"depth_mm":0},
        ]
    },
    "Ragi": {
        "total_days": 110,
        "category": "Cereal",
        "season": "Kharif",
        "water": "Low",
        "stages": [
            {"name":"Germination","start":0,"end":10,"water_need":"Low","ndvi_range":[0.1, 0.25],"irrigation_days":5,"depth_mm":15},
            {"name":"Seedling","start":10,"end":25,"water_need":"Low","ndvi_range":[0.2, 0.4],"irrigation_days":7,"depth_mm":20},
            {"name":"Tillering","start":25,"end":55,"water_need":"Moderate","ndvi_range":[0.35, 0.6],"irrigation_days":10,"depth_mm":20},
            {"name":"Panicle Initiation","start":55,"end":75,"water_need":"High","ndvi_range":[0.45, 0.65],"irrigation_days":7,"depth_mm":25},
            {"name":"Flowering","start":75,"end":90,"water_need":"Critical","ndvi_range":[0.5, 0.7],"irrigation_days":5,"depth_mm":20},
            {"name":"Grain Fill","start":90,"end":110,"water_need":"Low","ndvi_range":[0.3, 0.5],"irrigation_days":12,"depth_mm":10},
        ]
    },
    "Maize": {
        "total_days": 100,
        "category": "Cereal",
        "season": "Kharif/Rabi",
        "water": "Moderate",
        "stages": [
            {"name":"Germination","start":0,"end":10,"water_need":"Low","ndvi_range":[0.1, 0.25],"irrigation_days":5,"depth_mm":20},
            {"name":"Vegetative","start":10,"end":40,"water_need":"Moderate","ndvi_range":[0.3, 0.6],"irrigation_days":8,"depth_mm":25},
            {"name":"Tasseling","start":40,"end":55,"water_need":"Critical","ndvi_range":[0.55, 0.8],"irrigation_days":5,"depth_mm":30},
            {"name":"Silking","start":55,"end":70,"water_need":"Critical","ndvi_range":[0.6, 0.85],"irrigation_days":4,"depth_mm":30},
            {"name":"Grain Fill","start":70,"end":90,"water_need":"Moderate","ndvi_range":[0.4, 0.65],"irrigation_days":7,"depth_mm":20},
            {"name":"Maturity","start":90,"end":100,"water_need":"Low","ndvi_range":[0.2, 0.4],"irrigation_days":14,"depth_mm":0},
        ]
    },
    "Jowar": {
        "total_days": 115,
        "category": "Cereal",
        "season": "Kharif/Rabi",
        "water": "Low",
        "stages": [
            {"name":"Germination","start":0,"end":8,"water_need":"Low","ndvi_range":[0.1, 0.22],"irrigation_days":5,"depth_mm":15},
            {"name":"Seedling","start":8,"end":25,"water_need":"Low","ndvi_range":[0.2, 0.4],"irrigation_days":8,"depth_mm":20},
            {"name":"Vegetative","start":25,"end":60,"water_need":"Moderate","ndvi_range":[0.35, 0.62],"irrigation_days":10,"depth_mm":25},
            {"name":"Flowering","start":60,"end":80,"water_need":"Critical","ndvi_range":[0.5, 0.7],"irrigation_days":7,"depth_mm":25},
            {"name":"Grain Fill","start":80,"end":100,"water_need":"Moderate","ndvi_range":[0.4, 0.6],"irrigation_days":10,"depth_mm":15},
            {"name":"Maturity","start":100,"end":115,"water_need":"Low","ndvi_range":[0.2, 0.4],"irrigation_days":14,"depth_mm":0},
        ]
    },
    "Bajra": {
        "total_days": 85,
        "category": "Cereal",
        "season": "Kharif",
        "water": "Very Low",
        "stages": [
            {"name":"Germination","start":0,"end":7,"water_need":"Low","ndvi_range":[0.1, 0.2],"irrigation_days":5,"depth_mm":12},
            {"name":"Vegetative","start":7,"end":35,"water_need":"Low","ndvi_range":[0.25, 0.5],"irrigation_days":10,"depth_mm":20},
            {"name":"Flowering","start":35,"end":55,"water_need":"Critical","ndvi_range":[0.45, 0.65],"irrigation_days":7,"depth_mm":22},
            {"name":"Grain Fill","start":55,"end":75,"water_need":"Moderate","ndvi_range":[0.35, 0.55],"irrigation_days":10,"depth_mm":15},
            {"name":"Maturity","start":75,"end":85,"water_need":"Low","ndvi_range":[0.2, 0.38],"irrigation_days":14,"depth_mm":0},
        ]
    },
    "Toor Dal": {
        "total_days": 165,
        "category": "Pulse",
        "season": "Kharif",
        "water": "Low",
        "stages": [
            {"name":"Germination","start":0,"end":10,"water_need":"Low","ndvi_range":[0.1, 0.25],"irrigation_days":7,"depth_mm":15},
            {"name":"Vegetative","start":10,"end":60,"water_need":"Low","ndvi_range":[0.25, 0.5],"irrigation_days":12,"depth_mm":20},
            {"name":"Flowering","start":60,"end":90,"water_need":"High","ndvi_range":[0.4, 0.62],"irrigation_days":8,"depth_mm":25},
            {"name":"Pod Fill","start":90,"end":140,"water_need":"Moderate","ndvi_range":[0.35, 0.58],"irrigation_days":10,"depth_mm":20},
            {"name":"Maturity","start":140,"end":165,"water_need":"Low","ndvi_range":[0.2, 0.4],"irrigation_days":15,"depth_mm":0},
        ]
    },
    "Chickpea": {
        "total_days": 100,
        "category": "Pulse",
        "season": "Rabi",
        "water": "Very Low",
        "stages": [
            {"name":"Germination","start":0,"end":10,"water_need":"Low","ndvi_range":[0.1, 0.22],"irrigation_days":7,"depth_mm":15},
            {"name":"Vegetative","start":10,"end":40,"water_need":"Low","ndvi_range":[0.25, 0.48],"irrigation_days":12,"depth_mm":18},
            {"name":"Flowering","start":40,"end":60,"water_need":"Critical","ndvi_range":[0.38, 0.58],"irrigation_days":8,"depth_mm":22},
            {"name":"Pod Fill","start":60,"end":85,"water_need":"Moderate","ndvi_range":[0.3, 0.52],"irrigation_days":10,"depth_mm":18},
            {"name":"Maturity","start":85,"end":100,"water_need":"Low","ndvi_range":[0.18, 0.38],"irrigation_days":14,"depth_mm":0},
        ]
    },
    "Green Gram": {
        "total_days": 65,
        "category": "Pulse",
        "season": "Kharif/Rabi",
        "water": "Low",
        "stages": [
            {"name":"Germination","start":0,"end":7,"water_need":"Low","ndvi_range":[0.1, 0.22],"irrigation_days":5,"depth_mm":12},
            {"name":"Vegetative","start":7,"end":30,"water_need":"Moderate","ndvi_range":[0.28, 0.52],"irrigation_days":8,"depth_mm":18},
            {"name":"Flowering","start":30,"end":45,"water_need":"Critical","ndvi_range":[0.4, 0.58],"irrigation_days":6,"depth_mm":22},
            {"name":"Pod Fill","start":45,"end":60,"water_need":"Moderate","ndvi_range":[0.3, 0.5],"irrigation_days":8,"depth_mm":15},
            {"name":"Maturity","start":60,"end":65,"water_need":"Low","ndvi_range":[0.18, 0.35],"irrigation_days":12,"depth_mm":0},
        ]
    },
    "Black Gram": {
        "total_days": 75,
        "category": "Pulse",
        "season": "Kharif/Rabi",
        "water": "Low",
        "stages": [
            {"name":"Germination","start":0,"end":7,"water_need":"Low","ndvi_range":[0.1, 0.22],"irrigation_days":5,"depth_mm":12},
            {"name":"Vegetative","start":7,"end":30,"water_need":"Moderate","ndvi_range":[0.25, 0.5],"irrigation_days":8,"depth_mm":18},
            {"name":"Flowering","start":30,"end":50,"water_need":"Critical","ndvi_range":[0.38, 0.56],"irrigation_days":6,"depth_mm":20},
            {"name":"Pod Fill","start":50,"end":68,"water_need":"Moderate","ndvi_range":[0.28, 0.48],"irrigation_days":8,"depth_mm":15},
            {"name":"Maturity","start":68,"end":75,"water_need":"Low","ndvi_range":[0.15, 0.32],"irrigation_days":12,"depth_mm":0},
        ]
    },
    "Groundnut": {
        "total_days": 125,
        "category": "Oilseed",
        "season": "Kharif/Rabi",
        "water": "Moderate",
        "stages": [
            {"name":"Germination","start":0,"end":10,"water_need":"Moderate","ndvi_range":[0.1, 0.25],"irrigation_days":5,"depth_mm":18},
            {"name":"Vegetative","start":10,"end":35,"water_need":"Moderate","ndvi_range":[0.28, 0.55],"irrigation_days":8,"depth_mm":22},
            {"name":"Flowering","start":35,"end":55,"water_need":"Critical","ndvi_range":[0.42, 0.65],"irrigation_days":6,"depth_mm":25},
            {"name":"Pegging","start":55,"end":75,"water_need":"Critical","ndvi_range":[0.45, 0.68],"irrigation_days":6,"depth_mm":25},
            {"name":"Pod Fill","start":75,"end":105,"water_need":"High","ndvi_range":[0.38, 0.6],"irrigation_days":8,"depth_mm":22},
            {"name":"Maturity","start":105,"end":125,"water_need":"Low","ndvi_range":[0.22, 0.42],"irrigation_days":14,"depth_mm":0},
        ]
    },
    "Sunflower": {
        "total_days": 95,
        "category": "Oilseed",
        "season": "Kharif/Rabi",
        "water": "Moderate",
        "stages": [
            {"name":"Germination","start":0,"end":8,"water_need":"Low","ndvi_range":[0.1, 0.22],"irrigation_days":5,"depth_mm":15},
            {"name":"Vegetative","start":8,"end":35,"water_need":"Moderate","ndvi_range":[0.3, 0.58],"irrigation_days":8,"depth_mm":22},
            {"name":"Bud Formation","start":35,"end":55,"water_need":"High","ndvi_range":[0.48, 0.72],"irrigation_days":6,"depth_mm":28},
            {"name":"Flowering","start":55,"end":70,"water_need":"Critical","ndvi_range":[0.52, 0.78],"irrigation_days":5,"depth_mm":28},
            {"name":"Seed Fill","start":70,"end":85,"water_need":"High","ndvi_range":[0.38, 0.62],"irrigation_days":7,"depth_mm":22},
            {"name":"Maturity","start":85,"end":95,"water_need":"Low","ndvi_range":[0.2, 0.38],"irrigation_days":14,"depth_mm":0},
        ]
    },
    "Soybean": {
        "total_days": 100,
        "category": "Oilseed",
        "season": "Kharif",
        "water": "Moderate",
        "stages": [
            {"name":"Germination","start":0,"end":8,"water_need":"Moderate","ndvi_range":[0.1, 0.25],"irrigation_days":5,"depth_mm":18},
            {"name":"Vegetative","start":8,"end":35,"water_need":"Moderate","ndvi_range":[0.32, 0.62],"irrigation_days":8,"depth_mm":22},
            {"name":"Flowering","start":35,"end":55,"water_need":"Critical","ndvi_range":[0.5, 0.75],"irrigation_days":6,"depth_mm":28},
            {"name":"Pod Fill","start":55,"end":80,"water_need":"Critical","ndvi_range":[0.52, 0.78],"irrigation_days":6,"depth_mm":28},
            {"name":"Maturity","start":80,"end":100,"water_need":"Low","ndvi_range":[0.25, 0.48],"irrigation_days":12,"depth_mm":0},
        ]
    },
    "Sugarcane": {
        "total_days": 330,
        "category": "Commercial",
        "season": "Annual",
        "water": "Very High",
        "stages": [
            {"name":"Germination","start":0,"end":35,"water_need":"High","ndvi_range":[0.15, 0.4],"irrigation_days":5,"depth_mm":30},
            {"name":"Tillering","start":35,"end":120,"water_need":"High","ndvi_range":[0.4, 0.65],"irrigation_days":7,"depth_mm":35},
            {"name":"Grand Growth","start":120,"end":270,"water_need":"Critical","ndvi_range":[0.6, 0.85],"irrigation_days":7,"depth_mm":40},
            {"name":"Maturation","start":270,"end":330,"water_need":"Low","ndvi_range":[0.45, 0.7],"irrigation_days":14,"depth_mm":15},
        ]
    },
    "Cotton": {
        "total_days": 185,
        "category": "Commercial",
        "season": "Kharif",
        "water": "Moderate",
        "stages": [
            {"name":"Germination","start":0,"end":10,"water_need":"Moderate","ndvi_range":[0.1, 0.25],"irrigation_days":5,"depth_mm":20},
            {"name":"Seedling","start":10,"end":30,"water_need":"Moderate","ndvi_range":[0.22, 0.45],"irrigation_days":8,"depth_mm":22},
            {"name":"Squaring","start":30,"end":60,"water_need":"High","ndvi_range":[0.38, 0.62],"irrigation_days":7,"depth_mm":28},
            {"name":"Flowering","start":60,"end":90,"water_need":"Critical","ndvi_range":[0.48, 0.72],"irrigation_days":6,"depth_mm":30},
            {"name":"Boll Development","start":90,"end":140,"water_need":"High","ndvi_range":[0.42, 0.68],"irrigation_days":7,"depth_mm":25},
            {"name":"Boll Opening","start":140,"end":185,"water_need":"Low","ndvi_range":[0.25, 0.48],"irrigation_days":14,"depth_mm":10},
        ]
    },
    "Coffee": {
        "total_days": 365,
        "category": "Plantation",
        "season": "Perennial",
        "water": "High",
        "stages": [
            {"name":"Flowering","start":0,"end":60,"water_need":"Critical","ndvi_range":[0.55, 0.75],"irrigation_days":5,"depth_mm":30},
            {"name":"Berry Development","start":60,"end":200,"water_need":"High","ndvi_range":[0.58, 0.78],"irrigation_days":7,"depth_mm":35},
            {"name":"Berry Ripening","start":200,"end":300,"water_need":"Moderate","ndvi_range":[0.52, 0.72],"irrigation_days":10,"depth_mm":25},
            {"name":"Harvesting & Rest","start":300,"end":365,"water_need":"Low","ndvi_range":[0.48, 0.68],"irrigation_days":14,"depth_mm":15},
        ]
    },
    "Coconut": {
        "total_days": 365,
        "category": "Plantation",
        "season": "Perennial",
        "water": "High",
        "stages": [
            {"name":"Vegetative","start":0,"end":120,"water_need":"High","ndvi_range":[0.6, 0.8],"irrigation_days":4,"depth_mm":35},
            {"name":"Flowering","start":120,"end":200,"water_need":"Critical","ndvi_range":[0.62, 0.82],"irrigation_days":4,"depth_mm":40},
            {"name":"Nut Development","start":200,"end":310,"water_need":"High","ndvi_range":[0.6, 0.8],"irrigation_days":5,"depth_mm":35},
            {"name":"Maturity","start":310,"end":365,"water_need":"Moderate","ndvi_range":[0.58, 0.78],"irrigation_days":7,"depth_mm":25},
        ]
    },
    "Arecanut": {
        "total_days": 365,
        "category": "Plantation",
        "season": "Perennial",
        "water": "High",
        "stages": [
            {"name":"Vegetative","start":0,"end":90,"water_need":"High","ndvi_range":[0.62, 0.82],"irrigation_days":3,"depth_mm":30},
            {"name":"Inflorescence","start":90,"end":180,"water_need":"Critical","ndvi_range":[0.65, 0.85],"irrigation_days":3,"depth_mm":35},
            {"name":"Fruit Development","start":180,"end":300,"water_need":"High","ndvi_range":[0.62, 0.82],"irrigation_days":4,"depth_mm":30},
            {"name":"Maturity","start":300,"end":365,"water_need":"Moderate","ndvi_range":[0.58, 0.78],"irrigation_days":6,"depth_mm":20},
        ]
    },
    "Mango": {
        "total_days": 365,
        "category": "Fruit",
        "season": "Perennial",
        "water": "Moderate",
        "stages": [
            {"name":"Vegetative Growth","start":0,"end":90,"water_need":"Moderate","ndvi_range":[0.55, 0.75],"irrigation_days":7,"depth_mm":25},
            {"name":"Flowering","start":90,"end":130,"water_need":"Critical","ndvi_range":[0.58, 0.78],"irrigation_days":5,"depth_mm":30},
            {"name":"Fruit Set","start":130,"end":200,"water_need":"High","ndvi_range":[0.6, 0.8],"irrigation_days":6,"depth_mm":30},
            {"name":"Fruit Development","start":200,"end":320,"water_need":"High","ndvi_range":[0.58, 0.78],"irrigation_days":7,"depth_mm":28},
            {"name":"Maturity","start":320,"end":365,"water_need":"Low","ndvi_range":[0.52, 0.72],"irrigation_days":12,"depth_mm":15},
        ]
    },
    "Banana": {
        "total_days": 300,
        "category": "Fruit",
        "season": "Annual/Perennial",
        "water": "Very High",
        "stages": [
            {"name":"Establishment","start":0,"end":40,"water_need":"High","ndvi_range":[0.25, 0.5],"irrigation_days":4,"depth_mm":35},
            {"name":"Vegetative","start":40,"end":150,"water_need":"High","ndvi_range":[0.48, 0.72],"irrigation_days":4,"depth_mm":40},
            {"name":"Shooting","start":150,"end":210,"water_need":"Critical","ndvi_range":[0.55, 0.78],"irrigation_days":3,"depth_mm":40},
            {"name":"Bunch Development","start":210,"end":270,"water_need":"Critical","ndvi_range":[0.55, 0.8],"irrigation_days":4,"depth_mm":38},
            {"name":"Maturity","start":270,"end":300,"water_need":"Moderate","ndvi_range":[0.48, 0.72],"irrigation_days":7,"depth_mm":25},
        ]
    },
    "Grapes": {
        "total_days": 180,
        "category": "Fruit",
        "season": "Semi-perennial",
        "water": "Moderate",
        "stages": [
            {"name":"Bud Break","start":0,"end":20,"water_need":"Moderate","ndvi_range":[0.15, 0.38],"irrigation_days":5,"depth_mm":22},
            {"name":"Shoot Growth","start":20,"end":60,"water_need":"High","ndvi_range":[0.38, 0.62],"irrigation_days":5,"depth_mm":28},
            {"name":"Flowering","start":60,"end":80,"water_need":"Critical","ndvi_range":[0.45, 0.68],"irrigation_days":4,"depth_mm":28},
            {"name":"Berry Development","start":80,"end":130,"water_need":"High","ndvi_range":[0.48, 0.72],"irrigation_days":5,"depth_mm":30},
            {"name":"Veraison","start":130,"end":155,"water_need":"Moderate","ndvi_range":[0.42, 0.65],"irrigation_days":7,"depth_mm":20},
            {"name":"Harvest","start":155,"end":180,"water_need":"Low","ndvi_range":[0.35, 0.55],"irrigation_days":10,"depth_mm":12},
        ]
    },
    "Pomegranate": {
        "total_days": 180,
        "category": "Fruit",
        "season": "Semi-perennial",
        "water": "Moderate",
        "stages": [
            {"name":"Dormancy Break","start":0,"end":20,"water_need":"Low","ndvi_range":[0.15, 0.35],"irrigation_days":7,"depth_mm":18},
            {"name":"Vegetative","start":20,"end":60,"water_need":"Moderate","ndvi_range":[0.32, 0.58],"irrigation_days":6,"depth_mm":22},
            {"name":"Flowering","start":60,"end":80,"water_need":"Critical","ndvi_range":[0.4, 0.65],"irrigation_days":5,"depth_mm":25},
            {"name":"Fruit Set","start":80,"end":120,"water_need":"High","ndvi_range":[0.42, 0.68],"irrigation_days":5,"depth_mm":25},
            {"name":"Fruit Development","start":120,"end":160,"water_need":"High","ndvi_range":[0.4, 0.65],"irrigation_days":6,"depth_mm":22},
            {"name":"Maturity","start":160,"end":180,"water_need":"Low","ndvi_range":[0.3, 0.52],"irrigation_days":10,"depth_mm":12},
        ]
    },
    "Tomato": {
        "total_days": 90,
        "category": "Vegetable",
        "season": "Kharif/Rabi",
        "water": "High",
        "stages": [
            {"name":"Transplanting","start":0,"end":15,"water_need":"High","ndvi_range":[0.15, 0.35],"irrigation_days":3,"depth_mm":20},
            {"name":"Vegetative","start":15,"end":40,"water_need":"High","ndvi_range":[0.3, 0.58],"irrigation_days":4,"depth_mm":25},
            {"name":"Flowering","start":40,"end":55,"water_need":"Critical","ndvi_range":[0.42, 0.68],"irrigation_days":3,"depth_mm":28},
            {"name":"Fruit Set","start":55,"end":70,"water_need":"Critical","ndvi_range":[0.45, 0.72],"irrigation_days":3,"depth_mm":28},
            {"name":"Fruit Ripening","start":70,"end":90,"water_need":"Moderate","ndvi_range":[0.32, 0.55],"irrigation_days":5,"depth_mm":18},
        ]
    },
    "Onion": {
        "total_days": 110,
        "category": "Vegetable",
        "season": "Kharif/Rabi",
        "water": "Moderate",
        "stages": [
            {"name":"Germination","start":0,"end":10,"water_need":"Moderate","ndvi_range":[0.1, 0.22],"irrigation_days":4,"depth_mm":15},
            {"name":"Seedling","start":10,"end":30,"water_need":"Moderate","ndvi_range":[0.2, 0.4],"irrigation_days":5,"depth_mm":18},
            {"name":"Bulb Initiation","start":30,"end":60,"water_need":"High","ndvi_range":[0.3, 0.52],"irrigation_days":5,"depth_mm":22},
            {"name":"Bulb Development","start":60,"end":90,"water_need":"Critical","ndvi_range":[0.35, 0.58],"irrigation_days":4,"depth_mm":22},
            {"name":"Maturity","start":90,"end":110,"water_need":"Low","ndvi_range":[0.2, 0.38],"irrigation_days":10,"depth_mm":0},
        ]
    },
    "Potato": {
        "total_days": 100,
        "category": "Vegetable",
        "season": "Rabi",
        "water": "High",
        "stages": [
            {"name":"Emergence","start":0,"end":15,"water_need":"Moderate","ndvi_range":[0.12, 0.28],"irrigation_days":5,"depth_mm":20},
            {"name":"Vegetative","start":15,"end":40,"water_need":"High","ndvi_range":[0.35, 0.65],"irrigation_days":5,"depth_mm":28},
            {"name":"Tuber Initiation","start":40,"end":60,"water_need":"Critical","ndvi_range":[0.48, 0.75],"irrigation_days":4,"depth_mm":30},
            {"name":"Tuber Bulking","start":60,"end":80,"water_need":"Critical","ndvi_range":[0.5, 0.78],"irrigation_days":4,"depth_mm":30},
            {"name":"Maturation","start":80,"end":100,"water_need":"Low","ndvi_range":[0.25, 0.48],"irrigation_days":10,"depth_mm":0},
        ]
    },
    "Chilli": {
        "total_days": 150,
        "category": "Vegetable/Spice",
        "season": "Kharif/Rabi",
        "water": "Moderate",
        "stages": [
            {"name":"Nursery","start":0,"end":30,"water_need":"Moderate","ndvi_range":[0.15, 0.35],"irrigation_days":4,"depth_mm":15},
            {"name":"Vegetative","start":30,"end":70,"water_need":"Moderate","ndvi_range":[0.28, 0.55],"irrigation_days":6,"depth_mm":20},
            {"name":"Flowering","start":70,"end":95,"water_need":"Critical","ndvi_range":[0.38, 0.62],"irrigation_days":5,"depth_mm":22},
            {"name":"Fruit Set","start":95,"end":120,"water_need":"High","ndvi_range":[0.4, 0.65],"irrigation_days":5,"depth_mm":22},
            {"name":"Fruit Development","start":120,"end":150,"water_need":"Moderate","ndvi_range":[0.35, 0.58],"irrigation_days":7,"depth_mm":18},
        ]
    },
    "Brinjal": {
        "total_days": 120,
        "category": "Vegetable",
        "season": "Kharif/Rabi",
        "water": "Moderate",
        "stages": [
            {"name":"Transplanting","start":0,"end":15,"water_need":"High","ndvi_range":[0.15, 0.32],"irrigation_days":3,"depth_mm":18},
            {"name":"Vegetative","start":15,"end":45,"water_need":"Moderate","ndvi_range":[0.28, 0.55],"irrigation_days":6,"depth_mm":20},
            {"name":"Flowering","start":45,"end":70,"water_need":"Critical","ndvi_range":[0.38, 0.62],"irrigation_days":5,"depth_mm":22},
            {"name":"Fruiting","start":70,"end":100,"water_need":"High","ndvi_range":[0.42, 0.68],"irrigation_days":5,"depth_mm":22},
            {"name":"Harvest","start":100,"end":120,"water_need":"Moderate","ndvi_range":[0.35, 0.58],"irrigation_days":7,"depth_mm":15},
        ]
    },
    "Cabbage": {
        "total_days": 90,
        "category": "Vegetable",
        "season": "Rabi",
        "water": "High",
        "stages": [
            {"name":"Transplanting","start":0,"end":15,"water_need":"High","ndvi_range":[0.15, 0.35],"irrigation_days":3,"depth_mm":18},
            {"name":"Leaf Formation","start":15,"end":45,"water_need":"High","ndvi_range":[0.3, 0.58],"irrigation_days":4,"depth_mm":22},
            {"name":"Head Formation","start":45,"end":70,"water_need":"Critical","ndvi_range":[0.4, 0.65],"irrigation_days":4,"depth_mm":25},
            {"name":"Head Maturity","start":70,"end":90,"water_need":"Moderate","ndvi_range":[0.35, 0.58],"irrigation_days":6,"depth_mm":15},
        ]
    },
    "Cauliflower": {
        "total_days": 90,
        "category": "Vegetable",
        "season": "Rabi",
        "water": "High",
        "stages": [
            {"name":"Transplanting","start":0,"end":15,"water_need":"High","ndvi_range":[0.12, 0.3],"irrigation_days":3,"depth_mm":18},
            {"name":"Leaf Growth","start":15,"end":45,"water_need":"High","ndvi_range":[0.28, 0.55],"irrigation_days":4,"depth_mm":22},
            {"name":"Curd Formation","start":45,"end":70,"water_need":"Critical","ndvi_range":[0.38, 0.62],"irrigation_days":4,"depth_mm":25},
            {"name":"Maturity","start":70,"end":90,"water_need":"Moderate","ndvi_range":[0.3, 0.52],"irrigation_days":6,"depth_mm":15},
        ]
    },
    "Turmeric": {
        "total_days": 270,
        "category": "Spice",
        "season": "Kharif",
        "water": "High",
        "stages": [
            {"name":"Sprouting","start":0,"end":30,"water_need":"High","ndvi_range":[0.12, 0.32],"irrigation_days":5,"depth_mm":25},
            {"name":"Vegetative","start":30,"end":120,"water_need":"High","ndvi_range":[0.4, 0.68],"irrigation_days":6,"depth_mm":30},
            {"name":"Rhizome Development","start":120,"end":210,"water_need":"Critical","ndvi_range":[0.48, 0.75],"irrigation_days":6,"depth_mm":32},
            {"name":"Maturation","start":210,"end":270,"water_need":"Moderate","ndvi_range":[0.35, 0.62],"irrigation_days":10,"depth_mm":18},
        ]
    },
    "Ginger": {
        "total_days": 210,
        "category": "Spice",
        "season": "Kharif",
        "water": "High",
        "stages": [
            {"name":"Sprouting","start":0,"end":25,"water_need":"High","ndvi_range":[0.1, 0.28],"irrigation_days":5,"depth_mm":22},
            {"name":"Vegetative","start":25,"end":90,"water_need":"High","ndvi_range":[0.35, 0.65],"irrigation_days":5,"depth_mm":28},
            {"name":"Rhizome Bulking","start":90,"end":165,"water_need":"Critical","ndvi_range":[0.45, 0.72],"irrigation_days":5,"depth_mm":30},
            {"name":"Maturation","start":165,"end":210,"water_need":"Moderate","ndvi_range":[0.32, 0.58],"irrigation_days":10,"depth_mm":15},
        ]
    },
    "Black Pepper": {
        "total_days": 365,
        "category": "Spice",
        "season": "Perennial",
        "water": "High",
        "stages": [
            {"name":"Vegetative","start":0,"end":90,"water_need":"High","ndvi_range":[0.55, 0.78],"irrigation_days":4,"depth_mm":28},
            {"name":"Flowering","start":90,"end":150,"water_need":"Critical","ndvi_range":[0.58, 0.82],"irrigation_days":4,"depth_mm":32},
            {"name":"Berry Development","start":150,"end":280,"water_need":"High","ndvi_range":[0.6, 0.85],"irrigation_days":5,"depth_mm":30},
            {"name":"Maturity","start":280,"end":365,"water_need":"Moderate","ndvi_range":[0.52, 0.75],"irrigation_days":8,"depth_mm":20},
        ]
    },
}



def classify_crop(dates, series):
    """Phenology-based crop classifier — 31 Karnataka crops."""
    if not series or len(series) < 4:
        return {'crop': 'Unknown', 'confidence': 0, 'auto': False}
    # Deduplicate same-date readings
    by_date = {}
    for d, v in zip(dates, series):
        by_date.setdefault(d, []).append(v)
    dedup_dates  = sorted(by_date.keys())
    dedup_series = [sum(by_date[d]) / len(by_date[d]) for d in dedup_dates]
    # 3-point moving average
    smoothed = []
    for i in range(len(dedup_series)):
        lo, hi = max(0, i-1), min(len(dedup_series), i+2)
        smoothed.append(sum(dedup_series[lo:hi]) / (hi-lo))
    peak      = max(smoothed)
    peak_idx  = smoothed.index(peak)
    from datetime import datetime as dt
    d0         = dt.strptime(dedup_dates[0],       '%Y-%m-%d')
    d_last     = dt.strptime(dedup_dates[-1],       '%Y-%m-%d')
    d_peak     = dt.strptime(dedup_dates[peak_idx], '%Y-%m-%d')
    total_span = (d_last - d0).days
    days_since = (d_last - d_peak).days
    end_val    = sum(smoothed[-3:]) / min(3, len(smoothed))
    fall_rate  = (peak - end_val) / days_since if days_since > 0 else 0
    best, best_score = None, float('inf')
    for crop, sig in CROP_SIGNATURES.items():
        score = (abs(peak - sig['peak']) +
                 abs(total_span - sig['cycleDays']) / 100 +
                 abs(fall_rate - sig['fallRate']) * 50)
        if score < best_score:
            best_score, best = score, crop
    confidence = round(max(40, min(95, 100 - best_score * 30)))
    return {
        'crop': best, 'confidence': confidence, 'auto': True,
        'peak': round(peak, 3), 'spanDays': total_span,
        'category': CROP_CALENDARS.get(best, {}).get('category', ''),
        'season': CROP_CALENDARS.get(best, {}).get('season', ''),
    }


def get_crop_recommendation(scores: dict, smi_mean: float) -> list:
    """Return top 3 crops with scores, reasons, and irrigation requirement."""
    recs = []
    for crop, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]:
        recs.append({
            'crop': crop,
            'suitability': score,
            'category': CROP_CALENDARS.get(crop, {}).get('category', ''),
            'season': CROP_CALENDARS.get(crop, {}).get('season', ''),
            'reason': CROP_REASONS.get(crop, ''),
            'irrigation_need': CROP_IRRIGATION_NEED.get(crop, 'Moderate'),
        })
    return recs


def compute_all_suitabilities(sand_v, clay_v, ph_v, whc_v, drain_v, smi_v):
    """Compute suitability score 0-100 for all 31 crops."""
    return {
        crop: compute_soil_suitability(
            sand_v or 35, clay_v or 28, ph_v or 6.2,
            whc_v or 110, drain_v or 3, smi_v, **p)
        for crop, p in CROP_SOIL_PARAMS.items()
    }

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
        crop_params = CROP_SOIL_PARAMS
        suit_scores = compute_all_suitabilities(sand_v, clay_v, ph_v, whc_v, drain_v, smi_v)
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




@app.get("/crop-calendar")
def crop_calendar(crop: str = "Rice (Paddy)", day_of_season: int = 45):
    """
    Returns current growth stage + upcoming critical windows.
    Supports all 31 Karnataka crops.
    day_of_season: days since sowing (0 = sowing date).
    """
    try:
        # Fuzzy match crop name
        cal = CROP_CALENDARS.get(crop)
        if not cal:
            # Try partial match
            for key in CROP_CALENDARS:
                if crop.lower() in key.lower() or key.lower() in crop.lower():
                    crop = key
                    cal = CROP_CALENDARS[key]
                    break
        if not cal:
            available = list(CROP_CALENDARS.keys())
            raise HTTPException(
                status_code=404,
                detail=f"Crop '{crop}' not found. Available: {available}"
            )

        stages = cal["stages"]
        total  = cal["total_days"]
        day    = max(0, min(day_of_season, total))

        current_stage = stages[-1]
        for st in stages:
            if st["start"] <= day < st["end"]:
                current_stage = st
                break

        days_in_stage    = day - current_stage["start"]
        days_left_stage  = current_stage["end"] - day
        progress_pct     = round((day / total) * 100)
        days_to_harvest  = total - day

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

        irr_text = (
            f"Irrigate every {current_stage['irrigation_days']} days "
            f"with {current_stage['depth_mm']}mm depth"
            if current_stage["depth_mm"] > 0
            else "No irrigation needed — crop approaching maturity"
        )

        return {
            "status": "success",
            "crop": crop,
            "category": cal.get("category", ""),
            "season": cal.get("season", ""),
            "water_requirement": cal.get("water", ""),
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
            "data_source": "ICAR Karnataka crop calendar reference — 31 crops"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Crop calendar failed: {str(e)}")


@app.get("/crops/list")
def list_crops():
    """Return all 31 supported crops with category and season info."""
    return {
        "status": "success",
        "total": len(CROP_CALENDARS),
        "crops": [
            {
                "name": name,
                "category": data.get("category",""),
                "season": data.get("season",""),
                "water_requirement": data.get("water",""),
                "total_days": data.get("total_days", 0)
            }
            for name, data in CROP_CALENDARS.items()
        ]
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "message": "AeroCrop backend is running",
        "crops_supported": len(CROP_CALENDARS),
        "endpoints": ["/analyze", "/soil-intelligence", "/weather", "/crop-calendar", "/crops/list", "/administrative", "/health"]
    }
