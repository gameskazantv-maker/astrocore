from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class ChartRequest(BaseModel):
    date: str
    time: str
    lat: float
    lon: float
    timezone: str


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/ping")
def ping():
    return {"ping": "pong"}


@app.post("/chart")
def chart(data: ChartRequest):
    return {
        "meta": {
            "date": data.date,
            "time": data.time,
            "lat": data.lat,
            "lon": data.lon,
            "timezone": data.timezone,
        },
        "planets": {
            "Sun": {"lon": 80.0, "sign": "Gemini"},
            "Moon": {"lon": 120.0, "sign": "Leo"},
            "Mercury": {"lon": 95.0, "sign": "Cancer"},
        },
        "houses": {
            "ASC": {"lon": 110.0, "sign": "Cancer"},
            "MC": {"lon": 20.0, "sign": "Aries"},
        },
        "aspects": [
            {
                "p1": "Sun",
                "p2": "Moon",
                "type": "trine",
                "orb": 1.2
            }
        ]
    }
