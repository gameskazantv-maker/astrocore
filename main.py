from fastapi import FastAPI
from pydantic import BaseModel
import swisseph as swe
import pytz
from datetime import datetime

app = FastAPI()

# --- helpers ---

SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer",
    "Leo", "Virgo", "Libra", "Scorpio",
    "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]

def sign_from_lon(lon: float) -> str:
    return SIGNS[int(lon // 30)]

PLANETS = {
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

# --- request model ---

class ChartRequest(BaseModel):
    date: str       # YYYY-MM-DD
    time: str       # HH:MM
    lat: float
    lon: float
    timezone: str

# --- endpoints ---

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/ping")
def ping():
    return {"ping": "pong"}

@app.post("/chart")
def chart(data: ChartRequest):
    tz = pytz.timezone(data.timezone)
    dt = datetime.strptime(
        f"{data.date} {data.time}",
        "%Y-%m-%d %H:%M"
    )
    dt = tz.localize(dt)
    dt_utc = dt.astimezone(pytz.utc)

    jd = swe.julday(
        dt_utc.year,
        dt_utc.month,
        dt_utc.day,
        dt_utc.hour + dt_utc.minute / 60
    )

    # planets
    planets = {}
    for name, pid in PLANETS.items():
        lon, lat, dist = swe.calc_ut(jd, pid)[0]
        planets[name] = {
            "lon": round(lon, 2),
            "sign": sign_from_lon(lon)
        }

    # houses
    houses_raw, ascmc = swe.houses(jd, data.lat, data.lon)
    houses = {}
    for i in range(12):
        lon = houses_raw[i]
        houses[str(i + 1)] = {
            "lon": round(lon, 2),
            "sign": sign_from_lon(lon)
        }

    asc = ascmc[0]
    mc = ascmc[1]

    return {
        "meta": {
            "date": data.date,
            "time": data.time,
            "lat": data.lat,
            "lon": data.lon,
            "timezone": data.timezone
        },
        "planets": planets,
        "houses": houses,
        "points": {
            "ASC": {
                "lon": round(asc, 2),
                "sign": sign_from_lon(asc)
            },
            "MC": {
                "lon": round(mc, 2),
                "sign": sign_from_lon(mc)
            }
        }
    }
