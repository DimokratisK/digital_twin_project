import requests


def geocode(city: str):
    """Return (lat, lon) for any city name."""
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": city, "count": 1}

    resp = requests.get(url, params=params).json()
    result = resp.get("results")

    if not result:
        raise ValueError(f"City '{city}' not found.")

    item = result[0]
    return item["latitude"], item["longitude"]
