import pandas as pd
from twin_core.data_ingestion.weather import (
    get_weather, 
    get_air_quality, 
    get_environment_bundle
)

lat = 37.9838
lon = 23.7275

weather = get_weather(lat, lon)
air = get_air_quality(lat, lon)
bundle = get_environment_bundle(lat, lon)

# Convert hourly sections to DataFrames
weather_df = pd.DataFrame(weather["hourly"])
air_df = pd.DataFrame(air["hourly"])
bundle_weather_df = pd.DataFrame(bundle["weather"]["hourly"])
bundle_air_df = pd.DataFrame(bundle["air_quality"]["hourly"])

print(weather_df.head())
print(air_df.head())


print(bundle_weather_df.head())
print(bundle_air_df.head())
