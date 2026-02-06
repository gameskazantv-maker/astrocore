# main.py
# AstroCore API — FastAPI + Swiss Ephemeris
# Python 3.11+

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from datetime import datetime, timezone, timedelta, tzinfo
from typing import Dict, Any, Tuple, List, Optional, Literal

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

# Set ephemeris path
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

# Default — Regiomontanus (horary)
DEFAULT_HSYS = os.getenv("HSYS", "R").upper()

# Longitude mode:
# E -> "east positive" (обычно как в геокодерах)
# W -> "west positive" (если нужно совпасть с некоторыми астросайтами)
DEFAULT_LON_MODE = os.getenv("LON_MODE", "E").upper()  # "E" or "W"


def get_git_sha() -> str | None:
    # Railway / GitHub / other CI env vars (try in order)
    for key in (
        "RAILWAY_GIT_COMMIT_SHA",
        "GITHUB_SHA",
        "COMMIT_SHA",
        "SOURCE_VERSION",
        "VERCEL_GIT_COMMIT_SHA",
    ):
        v = os.getenv(key)
        if v:
            return v.strip()

    # fallback: local dev only (if .git exists and git is installed)
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(BASE_DIR),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return sha
    except Exception:
        return None


GIT_SHA = get_git_sha()


# ======================================================
# UTILS
# ======================================================

def norm360(x: float) -> float:
    x = x % 360.0
    return x + 360.0 if x < 0 else x


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
    if not tz_str or not isinstance(tz_str, str):
        raise HTTPException(status_code=422, detail="timezone must be a non-empty string")

    tz_str = tz_str.strip()
    tz_str = TZ_ALIASES.get(tz_str, tz_str)

    if tz_str.upper() in ("UTC", "ETC/UTC"):
        return timezone.utc

    m = _OFFSET_RE.match(tz_str.replace(" ", ""))
    if m:
        sign = 1 if m.group(1) == "+" else -1
        hours = int(m.group(2))
        minutes = int(m.group(3) or "0")
        if hours > 14 or minutes > 59:
            raise HTTPException(status_code=422, detail=f"Invalid timezone offset: {tz_str}")
        return timezone(sign * timedelta(hours=hours, minutes=minutes))

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
    handling wrap at 360.
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
    cusps: list of 12 cusp longitudes (1..12) in degrees.
    """
    if len(cusps) != 12:
        raise ValueError("cusps must have length 12")

    for i in range(12):
        start = cusps[i]
        end = cusps[(i + 1) % 12]
        if in_arc_ccw(pl_lon, start, end):
            return i + 1

    return 1


def normalize_lon_input(lon: float, lon_mode: str) -> float:
    """
    Convert input longitude into Swiss Ephemeris expected convention (east positive).
    lon_mode:
      - "E": input is east-positive (standard) -> keep
      - "W": input is west-positive -> invert sign
    """
    lon_mode = (lon_mode or DEFAULT_LON_MODE).upper()
    if lon_mode not in ("E", "W"):
        raise HTTPException(status_code=422, detail=f"Invalid lon_mode: {lon_mode}. Use 'E' or 'W'.")
    return float(lon if lon_mode == "E" else -lon)


def ang_diff(a: float, b: float) -> float:
    """Minimal absolute angular difference 0..180"""
    d = abs(norm360(a) - norm360(b))
    return d if d <= 180 else 360.0 - d


# ======================================================
# EXTRA CALC (tabs)
# ======================================================

def calc_antiscia(planets: Dict[str, Any], cusps_deg: List[float]) -> Tuple[List[dict], dict]:
    """
    Antiscia axis: 0°Cap / 0°Cancer (solstitial axis).
    anti = 180 - lon
    contra = anti + 180
    """
    out: List[dict] = []
    for body, pdata in planets.items():
        lon = float(pdata["lon"])
        anti = norm360(180.0 - lon)
        contra = norm360(anti + 180.0)

        out.append({
            "body": body,
            "lon": round(lon, 6),
            "antiscia_lon": round(anti, 6),
            "contra_lon": round(contra, 6),
            "antiscia_pos": pretty_pos(anti),
            "contra_pos": pretty_pos(contra),
            "antiscia_house": house_from_cusps(anti, cusps_deg),
            "contra_house": house_from_cusps(contra, cusps_deg),
        })

    meta = {"count": len(out), "axis": "0Cap/0Can"}
    return out, meta


def calc_lots(planets: Dict[str, Any], angles: Dict[str, Any], cusps_deg: List[float]) -> Tuple[dict, dict]:
    """
    MVP: Lot of Fortune only.
    Day/Night (MVP): by Sun house (7..12 => day, 1..6 => night).
    Fortune:
      Day: ASC + Moon - Sun
      Night: ASC + Sun - Moon
    """
    if "Sun" not in planets or "Moon" not in planets:
        return {}, {"count": 0, "note": "Sun or Moon missing"}

    asc = float(angles["ASC"]["lon"])
    sun = float(planets["Sun"]["lon"])
    moon = float(planets["Moon"]["lon"])

    sun_house = house_from_cusps(sun, cusps_deg)
    is_day = sun_house >= 7  # MVP rule

    if is_day:
        fortuna = norm360(asc + moon - sun)
    else:
        fortuna = norm360(asc + sun - moon)

    lots = {
        "fortuna": {
            "lon": round(fortuna, 6),
            "pos": pretty_pos(fortuna),
            "house": house_from_cusps(fortuna, cusps_deg),
        }
    }
    meta = {"count": len(lots), "computed": list(lots.keys()), "is_day": is_day, "sun_house": sun_house}
    return lots, meta


# MVP fixed stars list (small). Longitudes are approximate tropical positions (simplified).
FIXED_STARS = [
    {"name": "Regulus", "lon": 150.20, "mag": 1.35},
    {"name": "Spica", "lon": 204.00, "mag": 0.98},
    {"name": "Aldebaran", "lon": 69.00, "mag": 0.85},
    {"name": "Antares", "lon": 249.50, "mag": 1.06},
    {"name": "Sirius", "lon": 104.00, "mag": -1.46},
    {"name": "Vega", "lon": 285.20, "mag": 0.03},
    {"name": "Arcturus", "lon": 204.30, "mag": -0.05},
    {"name": "Fomalhaut", "lon": 333.70, "mag": 1.16},
]


def calc_fixed_stars(planets: Dict[str, Any], angles: Dict[str, Any], max_mag: float, orb: float) -> Tuple[List[dict], dict]:
    """
    MVP: finds conjunction-ish hits within 'orb' degrees between stars and (planets + ASC/MC).
    """
    targets: Dict[str, float] = {k: float(v["lon"]) for k, v in planets.items()}
    targets.update({"ASC": float(angles["ASC"]["lon"]), "MC": float(angles["MC"]["lon"])})

    out: List[dict] = []
    for s in FIXED_STARS:
        if float(s["mag"]) > float(max_mag):
            continue

        hits = []
        star_lon = float(s["lon"])
        for tname, tlon in targets.items():
            d = ang_diff(star_lon, tlon)
            if d <= float(orb):
                hits.append({"target": tname, "delta": round(d, 4)})

        if hits:
            hits.sort(key=lambda x: x["delta"])
            out.append({
                "star": s["name"],
                "lon": round(star_lon, 6),
                "pos": pretty_pos(star_lon),
                "mag": float(s["mag"]),
                "hits": hits,
            })

    meta = {"count": len(out), "max_mag": float(max_mag), "orb": float(orb), "stars_scanned": len(FIXED_STARS)}
    return out, meta


TRANSLATION_ASPECTS = [
    ("Conjunction", 0),
    ("Sextile", 60),
    ("Square", 90),
    ("Trine", 120),
    ("Opposition", 180),
]


def delta_to_exact_aspect(lon_a: float, lon_b: float, exact_angle: float) -> float:
    d = ang_diff(lon_a, lon_b)
    return abs(d - exact_angle)


def is_applying_mvp(
    lon_a: float, spd_a: float, lon_b: float, spd_b: float,
    exact_angle: float, dt_hours: float = 6.0
) -> bool:
    """
    MVP applying test: compare delta now vs delta after dt_hours using linear motion lon += speed*dt.
    speed is deg/day in Swiss Ephemeris output -> convert to deg/hour.
    """
    # Swiss speed is deg/day
    a2 = norm360(lon_a + (spd_a / 24.0) * dt_hours)
    b2 = norm360(lon_b + (spd_b / 24.0) * dt_hours)

    d1 = delta_to_exact_aspect(lon_a, lon_b, exact_angle)
    d2 = delta_to_exact_aspect(a2, b2, exact_angle)
    return d2 < d1


def calc_translation_of_light(planets: Dict[str, Any], orb: float) -> Tuple[List[dict], dict]:
    """
    MVP: Moon translates light from B to C if Moon is applying to aspect with B and also applying to aspect with C.
    This is simplified and meant to feed the UI tab.
    """
    if "Moon" not in planets:
        return [], {"count": 0, "note": "Moon missing"}

    m_lon = float(planets["Moon"]["lon"])
    m_spd = float(planets["Moon"].get("speed", 0.0))

    bodies = [k for k in planets.keys() if k != "Moon"]
    candidates = []  # (body, aspect_name, aspect_angle, orb_delta)

    for b in bodies:
        b_lon = float(planets[b]["lon"])
        b_spd = float(planets[b].get("speed", 0.0))

        for aname, aangle in TRANSLATION_ASPECTS:
            d = delta_to_exact_aspect(m_lon, b_lon, aangle)
            if d <= float(orb) and is_applying_mvp(m_lon, m_spd, b_lon, b_spd, aangle):
                candidates.append((b, aname, aangle, d))

    candidates.sort(key=lambda x: x[3])

    out: List[dict] = []
    for i in range(len(candidates)):
        b, asp_b, ang_b, d_b = candidates[i]
        for j in range(i + 1, len(candidates)):
            c, asp_c, ang_c, d_c = candidates[j]
            if b == c:
                continue
            out.append({
                "translator": "Moon",
                "from": b,
                "to": c,
                "aspect_from": asp_b,
                "aspect_to": asp_c,
                "orb_from": round(d_b, 4),
                "orb_to": round(d_c, 4),
            })

    meta = {"count": len(out), "orb": float(orb), "dt_hours": 6.0, "note": "MVP translation_of_light"}
    return out, meta


# ======================================================
# SCHEMAS
# ======================================================

IncludeKey = Literal["aspects", "antiscia", "translation", "fixed_stars", "lots"]
LonMode = Literal["E", "W"]


class ChartRequest(BaseModel):
    date: str
    time: str
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    timezone: str

    hsys: str | None = None      # e.g. "R", "K", ...
    lon_mode: LonMode | None = None  # "E" (east+) or "W" (west+)

    # NEW (tabs)
    include: Optional[List[IncludeKey]] = None
    fixed_stars_max_mag: float = 3.0
    fixed_stars_orb: float = 1.0
    orb_translation: float = 1.5

    lot_system: str = "default"
    lot_names: Optional[List[str]] = None


class ChartResponse(BaseModel):
    meta: dict
    planets: dict
    houses: dict
    angles: dict
    aspects: list
    aspects_meta: dict

    # NEW optional sections
    antiscia: Optional[list] = None
    antiscia_meta: Optional[dict] = None

    lots: Optional[dict] = None
    lots_meta: Optional[dict] = None

    fixed_stars: Optional[list] = None
    fixed_stars_meta: Optional[dict] = None

    translation_of_light: Optional[list] = None
    translation_meta: Optional[dict] = None


# ======================================================
# CORE CALC
# ======================================================

def calc_houses(jd: float, lat: float, lon_east_pos: float, hsys: str) -> dict:
    try:
        hsys_b = (hsys or DEFAULT_HSYS).upper().encode("ascii")
        cusps_raw, ascmc = swe.houses(jd, lat, lon_east_pos, hsys_b)

        cusps_list = list(cusps_raw)
        if len(cusps_list) == 13:
            cusps_list = cusps_list[1:13]
        elif len(cusps_list) >= 12:
            cusps_list = cusps_list[:12]
        else:
            raise RuntimeError(f"Unexpected cusps length: {len(cusps_list)}")

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
            "hsys": (hsys or DEFAULT_HSYS).upper(),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Swiss Ephemeris houses error: {e}")


def calc_planets(jd: float, cusps_deg: List[float]) -> dict:
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

app = FastAPI(title="AstroCore API", version="1.0.8")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
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
        "version": "1.0.8",
        "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "ephe_path": str(EPHE_PATH),
        "ephe_exists": EPHE_PATH.exists(),
        "default_hsys": DEFAULT_HSYS,
        "default_lon_mode": DEFAULT_LON_MODE,
        "git_sha": GIT_SHA or None,  # ✅ not empty string
    }


@app.post("/chart", response_model=ChartResponse)
def chart(req: ChartRequest):
    dt = parse_datetime(req.date, req.time, req.timezone)
    jd = julday(dt)

    hsys = (req.hsys or DEFAULT_HSYS).upper()
    lon_mode = (req.lon_mode or DEFAULT_LON_MODE).upper()
    lon_east_pos = normalize_lon_input(req.lon, lon_mode)

    houses_data = calc_houses(jd, req.lat, lon_east_pos, hsys)
    planets = calc_planets(jd, houses_data["cusps_deg"])
    aspects = calc_aspects(planets, houses_data["angles"])

    include_used = req.include or ["aspects"]
    # remove duplicates while preserving order
    include_used = list(dict.fromkeys(include_used))

    resp: Dict[str, Any] = {
        "meta": {
            "datetime": dt.isoformat(),
            "jd": round(jd, 6),
            "timezone": req.timezone,
            "hsys": houses_data["hsys"],
            "lon_mode": lon_mode,
            "lon_input": req.lon,
            "lon_used": round(lon_east_pos, 6),
            "include_used": include_used,
        },
        "planets": planets,
        "houses": houses_data["houses"],
        "angles": houses_data["angles"],
        "aspects": aspects,
        "aspects_meta": {
            "count": len(aspects),
            "note": "Zero aspects is a valid astrological state",
            "aspects_used": [{"type": n, "symbol": s, "angle": a, "orb": o} for (n, s, a, o) in ASPECTS],
            "luminary_bonus": LUM_BONUS,
        },
    }

    # Optional sections
    if "antiscia" in include_used:
        antiscia, antiscia_meta = calc_antiscia(planets, houses_data["cusps_deg"])
        resp["antiscia"] = antiscia
        resp["antiscia_meta"] = antiscia_meta

    if "lots" in include_used:
        lots, lots_meta = calc_lots(planets, houses_data["angles"], houses_data["cusps_deg"])
        resp["lots"] = lots
        resp["lots_meta"] = lots_meta

    if "fixed_stars" in include_used:
        fixed_stars, fixed_stars_meta = calc_fixed_stars(
            planets,
            houses_data["angles"],
            max_mag=req.fixed_stars_max_mag,
            orb=req.fixed_stars_orb,
        )
        resp["fixed_stars"] = fixed_stars
        resp["fixed_stars_meta"] = fixed_stars_meta

    if "translation" in include_used:
        tol, tol_meta = calc_translation_of_light(planets, orb=req.orb_translation)
        resp["translation_of_light"] = tol
        resp["translation_meta"] = tol_meta

    return resp


# ======================================================
# LOCAL / RAILWAY RUN
# ======================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

