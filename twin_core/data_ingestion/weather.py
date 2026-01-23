import requests
from datetime import date, timedelta
from typing import Dict, Any


def get_weather(latitude: float, longitude: float) -> Dict[str, Any]:
    """Current weather conditions from Open-Meteo."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m,relative_humidity_2m,cloudcover,uv_index",
        "forecast_days": 1
    }

    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()


def get_air_quality(latitude: float, longitude: float) -> Dict[str, Any]:
    """Air quality data from Open-Meteo."""
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "pm10,pm2_5,ozone,nitrogen_dioxide,sulphur_dioxide"
    }

    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()


def get_environment_bundle(lat: float, lon: float) -> Dict[str, Any]:
    """Convenience wrapper for full environment context."""
    return {
        "weather": get_weather(lat, lon),
        "air_quality": get_air_quality(lat, lon),
    }
