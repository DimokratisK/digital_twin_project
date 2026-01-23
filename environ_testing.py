from twin_core.data_ingestion.weather import get_weather, get_air_quality, get_environment_bundle
from twin_core.utils.fix_path import fix_path

path = r"c:\Users\dimok\Downloads\PhD\Udemy_courses\Deep_Learning_for_medical_imaging\AI-IN-MEDICAL-MATERIALS_NEW\AI-IN-MEDICAL-MATERIALS\04-Pneumonia-Classification\rsna-pneumonia-detection-challenge\stage_2_train_images"
new_path = fix_path(path)
print(new_path)

# Example coordinates (latitude, longitude)
latitude = 37.9838   # Athens
longitude = 23.7275 

# Get current weather
weather_data = get_weather(latitude, longitude)
print("Weather data:")
print(weather_data)

# Get air quality
air_quality_data = get_air_quality(latitude, longitude)
print("\nAir quality data:")
print(air_quality_data)

# Get full environment bundle
env_bundle = get_environment_bundle(latitude, longitude)
print("\nFull environment bundle:")
print(env_bundle)