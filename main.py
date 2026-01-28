import os
from datetime import datetime
from typing import Dict

import pytz
import swisseph as swe
from fastapi import FastAPI
from pydantic import BaseModel


# =========================
# Swiss Ephemeris setup
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EPHE_PATH = os.path.join(BASE_DIR, "ephe")
swe.set_ephe_path(EPHE_PATH)

# Используем Swiss Ephemeris
swe.set_sid_mode(swe.SIDM_FAGAN_BRADLEY, 0, 0)


# =========================
# FastAPI app
# =========================

app = FastAPI(title="AstroCore API")


# =========================
# Models
# =========================

class ChartRequest(BaseModel):
    date: str        # YYYY-MM-DD
    time: str        # HH:MM
    lat: float
    lon: float
    timezone: str    # Europe/Moscow


# =========================
# Helpers
# =========================

SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer",
    "Leo", "Virgo", "Libra", "Scorpio",
    "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]


def lon_to_sign(lon: float) -> str:
    return SIGNS[int(lon // 30) % 12]


def to_julian(dt: datetime) -> float:
    return swe.julday(
        dt.year,
        dt.month,
        dt.day,
        dt.hour + dt.minute / 60.0
    )


# =========================
# Routes
# =========================

@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/ping")
def ping():
    return {"ping": "pong"}


@app.post("/chart")
def chart(req: ChartRequest):
    # --- Parse datetime ---
    tz = pytz.timezone(req.timezone)
    naive_dt = datetime.strptime(
        f"{req.date} {req.time}",
        "%Y-%m-%d %H:%M"
    )
    local_dt = tz.localize(naive_dt)
    utc_dt = local_dt.astimezone(pytz.UTC)

    jd = to_julian(utc_dt)

    # --- Planets ---
    planets: Dict[str, Dict] = {}

    planet_ids = {
        "Sun": swe.SUN,
        "Moon": swe.MOON,
        "Mercury": swe.MERCURY,
        "Venus": swe.VENUS,
        "Mars": swe.MARS,
        "Jupiter": swe.JUPITER,
        "Saturn": swe.SATURN,
        "Uranus": swe.URANUS,
        "Neptune": swe.NEPTUNE,
        "Pluto": swe.PLUTO,
    }

    for name, pid in planet_ids.items():
        lon, lat, dist, speed_lon = swe.calc_ut(jd, pid)[0]
        planets[name] = {
            "lon": round(lon, 3),
            "sign": lon_to_sign(lon)
        }

    # --- Houses (Placidus) ---
    houses_raw, ascmc = swe.houses(
        jd,
        req.lat,
        req.lon,
        b'P'
    )

    houses = {
        "ASC": {
            "lon": round(ascmc[0], 3),
            "sign": lon_to_sign(ascmc[0])
        },
        "MC": {
            "lon": round(ascmc[1], 3),
            "sign": lon_to_sign(ascmc[1])
        }
    }

    for i in range(12):
        houses[f"House_{i+1}"] = {
            "lon": round(houses_raw[i], 3),
            "sign": lon_to_sign(houses_raw[i])
        }

    # --- Response ---
    return {
        "meta": {
            "date": req.date,
            "time": req.time,
            "lat": req.lat,
            "lon": req.lon,
            "timezone": req.timezone
        },
        "planets": planets,
        "houses": houses
    }

