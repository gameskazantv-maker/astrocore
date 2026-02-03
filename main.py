# main.py
# AstroCore API — FastAPI + Swiss Ephemeris
# Python 3.11+

from __future__ import annotations

import os
import re
from pathlib import Path
from datetime import datetime, timezone, timedelta, tzinfo
from typing import Dict, Any, Tuple, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import swisseph as swe
import uvicorn


# ======================================================
# CONFIG
# ======================================================

BASE_DIR = Path(__file__).parent
EPHE_PATH = Path(os.getenv("EPHE_PATH", str(BASE_DIR / "ephe"))).resolve()

# Set ephemeris path (if folder exists — great; if not — still ok for many use-cases)
swe.set_ephe_path(str(EPHE_PATH))

PLANETS = [
    ("Sun", swe.SUN),
    ("Moon", swe.MOON),
    ("Mercury", swe.MERCURY),
    ("Venus", swe.VENUS),
    ("Mars", swe.MARS),
    ("Jupiter", swe.JUPITER),
    ("Saturn", swe.SATURN),
    ("Uranus", swe.URANUS),
    ("Neptune", swe.NEPTUNE),
    ("Pluto", swe.PLUTO),
]

SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces",
]
SIGN_SYMBOLS = ["♈", "♉", "♊", "♋", "♌", "♍", "♎", "♏", "♐", "♑", "♒", "♓"]

ASPECTS = [
    ("Conjunction", "☌", 0, 8),
    ("Opposition", "☍", 180, 8),
    ("Square", "□", 90, 6),
    ("Trine", "△", 120, 6),
    ("Sextile", "✶", 60, 4),
]

LUMINARIES = {"Sun", "Moon"}
LUM_BONUS = {"Conjunction": 2, "Opposition": 2, "Square": 1, "Trine": 1, "Sextile": 1}

# Optional: common abbreviations fallback (offset-based; no DST handling here)
TZ_ALIASES = {"MSK": "+03:00", "UTC": "UTC", "GMT": "UTC"}

# Default house system (horary default: Regiomontanus)
DEFAULT_HSYS = os.getenv("HSYS", "R").upper()


# ======================================================
# UTILS
# ======================================================

def norm360(x: float) -> float:
    return x % 360.0


def shortest_arc(a: float, b: float) -> float:
    d = abs(norm360(a) - norm360(b))
    return d if d <= 180 else 360 - d


def sign_info(lon: float) -> Tuple[str, str, float]:
    idx = int(norm360(lon) // 30)
    return SIGNS[idx], SIGN_SYMBOLS[idx], lon % 30


def dms(x: float) -> Tuple[int, int, int]:
    x = abs(x)
    degrees = int(x)
    minutes = int((x - degrees) * 60)
    seconds = round(((x - degrees) * 60 - minutes) * 60)

    if seconds == 60:
        seconds = 0
        minutes += 1
    if minutes == 60:
        minutes = 0
        degrees += 1

    return degrees, minutes, seconds


def pretty_pos(lon: float) -> dict:
    sign, sym, deg = sign_info(lon)
    d, m, s = dms(deg)
    return {
        "lon": round(norm360(lon), 6),
        "sign": sign,
        "symbol": sym,
        "deg_in_sign": round(deg, 4),
        "dms": {"deg": d, "min": m, "sec": s},
        "label": f"{d}°{m:02d}'{s:02d}\" {sym}",
    }


_OFFSET_RE = re.compile(r"^(?:UTC|GMT)?\s*([+-])\s*(\d{1,2})(?::?(\d{2}))?$", re.IGNORECASE)


def parse_tz(tz_str: str) -> tzinfo:
    """
    Accepts:
      - IANA: "Europe/Moscow"
      - "UTC"
      - Offsets: "+03:00", "-0530", "UTC+3", "GMT+03:00"
    """
    if not tz_str or not isinstance(tz_str, str):
        raise HTTPException(status_code=422, detail="timezone must be a non-empty string")

    tz_str = tz_str.strip()
    tz_str = TZ_ALIASES.get(tz_str, tz_str)

    if tz_str.upper() in ("UTC", "ETC/UTC"):
        return timezone.utc

    # Offset like +03:00 / UTC+3 / GMT-0530
    m = _OFFSET_RE.match(tz_str.replace(" ", ""))
    if m:
        sign = 1 if m.group(1) == "+" else -1
        hours = int(m.group(2))
        minutes = int(m.group(3) or "0")
        if hours > 14 or minutes > 59:
            raise HTTPException(status_code=422, detail=f"Invalid timezone offset: {tz_str}")
        return timezone(sign * timedelta(hours=hours, minutes=minutes))

    # IANA timezone
    try:
        return ZoneInfo(tz_str)
    except ZoneInfoNotFoundError:
        raise HTTPException(status_code=422, detail=f"Unknown timezone: {tz_str}")


def parse_datetime(date: str, time: str, tz: str) -> datetime:
    zone = parse_tz(tz)
    dt_str = f"{date} {time}"

    try:
        if time.count(":") == 2:
            dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        else:
            dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail="Invalid date/time format. Use date YYYY-MM-DD and time HH:MM or HH:MM:SS",
        )

    return dt.replace(tzinfo=zone)


def julday(dt: datetime) -> float:
    utc = dt.astimezone(timezone.utc)
    hour = utc.hour + utc.minute / 60.0 + utc.second / 3600.0
    return float(swe.julday(utc.year, utc.month, utc.day, hour))


def in_arc_ccw(x: float, start: float, end: float) -> bool:
    """
    True if longitude x is in CCW arc from start to end (increasing degrees),
    handling wrap at 360. start inclusive, end exclusive.
    """
    x = norm360(x)
    start = norm360(start)
    end = norm360(end)
    if start <= end:
        return start <= x < end
    return (x >= start) or (x < end)


def house_from_cusps(pl_lon: float, cusps: List[float]) -> int:
    """
    Determine house by which cusp interval contains the planet longitude.
    cusps: list of 12 cusp longitudes (House 1..12) in degrees.
    """
    if len(cusps) != 12:
        raise ValueError("cusps must have length 12")

    pl_lon = norm360(pl_lon)

    for i in range(12):
        start = float(cusps[i])
        end = float(cusps[(i + 1) % 12])
        if in_arc_ccw(pl_lon, start, end):
            return i + 1

    # Fallback (should not happen)
    return 1


def normalize_hsys(hsys: Optional[str]) -> str:
    """
    Swiss Ephemeris expects a single ASCII letter for house system.
    We'll accept strings like "R", "Regiomontanus", "Placidus", etc.
    """
    if not hsys:
        return DEFAULT_HSYS

    s = str(hsys).strip()
    if not s:
        return DEFAULT_HSYS

    # If they pass full name, take first letter as a pragmatic fallback
    # (e.g., "Regiomontanus" -> "R")
    letter = s[0].upper()

    # Swiss Ephemeris supported house system letters are multiple;
    # we won't hard-fail here — just ensure single ASCII char.
    try:
        letter.encode("ascii")
    except UnicodeEncodeError:
        raise HTTPException(status_code=422, detail=f"Invalid hsys (non-ascii): {hsys}")

    return letter


# ======================================================
# SCHEMAS
# ======================================================

class ChartRequest(BaseModel):
    date: str
    time: str
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    timezone: str
    hsys: Optional[str] = None  # e.g. "R", "P", "K", ...


class ChartResponse(BaseModel):
    meta: dict
    planets: dict
    houses: dict
    angles: dict
    aspects: list
    aspects_meta: dict


# ======================================================
# CORE CALC
# ======================================================

def calc_houses(jd: float, lat: float, lon: float, hsys: str) -> dict:
    """
    Houses and angles from Swiss Ephemeris.
    Returns formatted houses + angles and raw cusp degrees for planet house assignment.
    """
    try:
        hsys_letter = normalize_hsys(hsys)
        hsys_b = hsys_letter.encode("ascii")

        # Swiss Ephemeris houses: (jd_ut, lat, lon, hsys)
        cusps_raw, ascmc = swe.houses(jd, lat, lon, hsys_b)

        # Normalize cusps to exactly 12 items (House 1..12)
        cusps_list = list(cusps_raw)
        if len(cusps_list) == 13:
            cusps_list = cusps_list[1:13]  # drop dummy 0 index
        elif len(cusps_list) >= 12:
            cusps_list = cusps_list[:12]
        else:
            raise RuntimeError(f"Unexpected cusps length: {len(cusps_list)}")

        # ascmc expected at least [ASC, MC, ...]
        if len(ascmc) < 2:
            raise RuntimeError(f"Unexpected ascmc length: {len(ascmc)}")

        asc = float(ascmc[0])
        mc = float(ascmc[1])

        houses = {str(i): pretty_pos(float(cusp_lon)) for i, cusp_lon in enumerate(cusps_list, start=1)}
        angles = {
            "ASC": pretty_pos(asc),
            "DSC": pretty_pos(norm360(asc + 180.0)),
            "MC": pretty_pos(mc),
            "IC": pretty_pos(norm360(mc + 180.0)),
        }

        return {
            "houses": houses,
            "angles": angles,
            "cusps_deg": [float(x) for x in cusps_list],
            "hsys": hsys_letter,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Swiss Ephemeris houses error: {e}")


def calc_planets(jd: float, cusps_deg: List[float]) -> dict:
    """
    Planets from Swiss Ephemeris + house attribution using cusps intervals.
    """
    out: Dict[str, Any] = {}

    for name, pid in PLANETS:
        try:
            res, _ = swe.calc_ut(jd, pid, swe.FLG_SWIEPH | swe.FLG_SPEED)
            pl_lon = float(res[0])
            speed = float(res[3])

            house_num = house_from_cusps(pl_lon, cusps_deg)

            out[name] = {
                **pretty_pos(pl_lon),
                "speed": round(speed, 6),
                "is_retrograde": speed < 0,
                "house": house_num,
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Swiss Ephemeris error calculating {name}: {e}")

    return out


def calc_aspects(planets: Dict[str, Any], angles: Dict[str, Any]) -> list:
    points = {
        **{k: float(v["lon"]) for k, v in planets.items()},
        **{k: float(v["lon"]) for k, v in angles.items()},
    }

    aspects = []
    keys = list(points.keys())

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            d = shortest_arc(points[a], points[b])

            for name, sym, ang, orb in ASPECTS:
                bonus = LUM_BONUS.get(name, 0) if (a in LUMINARIES or b in LUMINARIES) else 0
                max_orb = orb + bonus
                delta = abs(d - ang)

                if delta <= max_orb:
                    aspects.append({
                        "a": a,
                        "b": b,
                        "type": name,
                        "symbol": sym,
                        "angle": ang,
                        "delta": round(d, 4),
                        "orb": round(delta, 4),
                        "orb_max": max_orb,
                        "bonus": bonus,
                    })

    aspects.sort(key=lambda x: x["orb"])
    return aspects


# ======================================================
# API
# ======================================================

app = FastAPI(title="AstroCore API", version="1.0.6")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # потом можно ограничить доменами
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    # НЕ падаем, если ephe отсутствует.
    if not EPHE_PATH.exists():
        print(f"[WARN] Ephemeris directory not found: {EPHE_PATH} (service will still run)")
        return

    se_files = list(EPHE_PATH.glob("*.se1"))
    if not se_files:
        print(f"[WARN] No ephemeris files (*.se1) found in {EPHE_PATH} (service will still run)")


@app.get("/")
def root():
    return {"service": "AstroCore API", "status": "running"}


@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/version")
def version():
    return {
        "version": "1.0.6",
        "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "ephe_path": str(EPHE_PATH),
        "ephe_exists": EPHE_PATH.exists(),
        "default_hsys": DEFAULT_HSYS,
        "git_sha": os.getenv("RAILWAY_GIT_COMMIT_SHA") or os.getenv("GIT_COMMIT") or "unknown",
    }


@app.post("/chart", response_model=ChartResponse)
def chart(req: ChartRequest):
    dt = parse_datetime(req.date, req.time, req.timezone)
    jd = julday(dt)

    hsys = normalize_hsys(req.hsys)

    houses_data = calc_houses(jd, req.lat, req.lon, hsys)
    planets = calc_planets(jd, houses_data["cusps_deg"])
    aspects = calc_aspects(planets, houses_data["angles"])

    return {
        "meta": {
            "datetime": dt.isoformat(),
            "jd": round(jd, 6),
            "timezone": req.timezone,
            "hsys": houses_data["hsys"],
        },
        "planets": planets,
        "houses": houses_data["houses"],
        "angles": houses_data["angles"],
        "aspects": aspects,
        "aspects_meta": {
            "count": len(aspects),
            "note": "Zero aspects is a valid astrological state",
            "aspects_used": [
                {"type": n, "symbol": s, "angle": a, "orb": o} for (n, s, a, o) in ASPECTS
            ],
            "luminary_bonus": LUM_BONUS,
        },
    }


# ======================================================
# LOCAL / RAILWAY RUN
# ======================================================

if __name__ == "__main__":
    # Railway provides PORT. Locally will use 8000.
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
