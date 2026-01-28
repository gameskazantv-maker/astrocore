import os
import math
from datetime import datetime
from typing import Dict

import pytz
import swisseph as swe
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# =========================
# Swiss Ephemeris setup
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EPHE_PATH = os.path.join(BASE_DIR, "ephe")

swe.set_ephe_path(EPHE_PATH)
swe.set_sid_mode(swe.SIDM_FAGAN_BRADLEY, 0, 0)


# =========================
# FastAPI app
# =========================

app = FastAPI(title="Astrocore API")


# =========================
# Models
# =========================

class ChartRequest(BaseModel):
    date: str        # YYYY-MM-DD
    time: str        # HH:MM
    lat: float
    lon: float
    timezone: str    # e.g. Europe/Moscow


# =========================
# Helpers
# =========================

SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]


def lon_to_sign(lon: float) -> str:
    return SIGNS[int(lon // 30) % 12]


def to_julian_day(date: str, time: str, tz_name: str) -> float:
    tz = pytz.timezone(tz_name)
    dt = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
    dt = tz.localize(dt).astimezone(pytz.utc)

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
    return {"status": "ok", "service": "astrocore"}


@app.get("/ping")
def ping():
    return {"pong": True}


@app.post("/chart")
def chart(req: ChartRequest):
    try:
        jd = to_julian_day(req.date, req.time, req.timezone)

        planets = {
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

        planets_out: Dict[str, Dict] = {}

        for name, pid in planets.items():
            lon, lat, dist = swe.calc_ut(jd, pid)[0][:3]
            lon = lon % 360

            planets_out[name] = {
                "lon": round(lon, 4),
                "sign": lon_to_sign(lon)
            }

        # Houses
        houses, ascmc = swe.houses(jd, req.lat, req.lon)

        houses_out = {
            "ASC": {
                "lon": round(ascmc[0] % 360, 4),
                "sign": lon_to_sign(ascmc[0] % 360)
            },
            "MC": {
                "lon": round(ascmc[1] % 360, 4),
                "sign": lon_to_sign(ascmc[1] % 360)
            }
        }

        return {
            "meta": {
                "date": req.date,
                "time": req.time,
                "lat": req.lat,
                "lon": req.lon,
                "timezone": req.timezone
            },
            "planets": planets_out,
            "houses": houses_out
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


