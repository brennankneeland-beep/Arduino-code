import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
import numpy as np
import pickle

cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": 48.4284,
    "longitude": -123.3656,
    "start_date": "2000-01-01",
    "end_date": "2025-12-31",
    "daily": [
        "temperature_2m_mean",
        "temperature_2m_max",
        "temperature_2m_min",
        "wind_speed_10m_max",
        "precipitation_sum"
    ]
}

response = openmeteo.weather_api(url, params=params)[0]
daily = response.Daily()

temps_mean = daily.Variables(0).ValuesAsNumpy()
temps_max  = daily.Variables(1).ValuesAsNumpy()
temps_min  = daily.Variables(2).ValuesAsNumpy()
wind_max   = daily.Variables(3).ValuesAsNumpy()
precip     = daily.Variables(4).ValuesAsNumpy()

dates = pd.date_range(
    start=pd.to_datetime(daily.Time(), unit="s", utc=True),
    end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
    freq=pd.Timedelta(seconds=daily.Interval()),
    inclusive="left"
)


min_len = min(
    len(dates),
    len(temps_mean),
    len(temps_max),
    len(temps_min),
    len(wind_max),
    len(precip)
)

dates = dates[:min_len]
temps_mean = temps_mean[:min_len]
temps_max = temps_max[:min_len]
temps_min = temps_min[:min_len]
wind_max = wind_max[:min_len]
precip = precip[:min_len]


X = []
y = []

for i in range(min_len - 1):
    day_of_year = dates[i].dayofyear

    X.append([
        day_of_year,
        temps_mean[i],
        temps_max[i],
        temps_min[i],
        wind_max[i],
        precip[i]
    ])

    if i < len(precip) - 1:
        y.append([1, 0] if precip[i+1] > 0 else [0, 1])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

mean = X.mean(axis=0)
std = X.std(axis=0)
std[std == 0] = 1

X = (X - mean) / std


scaler = {
    "mean": mean,
    "std": std
}

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)


with open("weather_inputs.pkl", "wb") as f:
    pickle.dump(X, f)

with open("weather_labels.pkl", "wb") as f:
    pickle.dump(y, f)


print("Samples:", len(X))
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Mean:", mean)
print("Std:", std)