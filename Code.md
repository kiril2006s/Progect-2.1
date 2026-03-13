import streamlit as st
import pandas as pd
import requests

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Функція отримання даних Open-Meteo
# -----------------------------

def get_weather_data(latitude, longitude, start_date, end_date):

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "rain_sum",
            "wind_speed_10m_max"
        ],
        "timezone": "auto"
    }

    response = requests.get(url, params=params)
    data = response.json()

    daily = data["daily"]

    df = pd.DataFrame({
        "date": daily["time"],
        "temp_max": daily["temperature_2m_max"],
        "temp_min": daily["temperature_2m_min"],
        "precipitation_sum": daily["precipitation_sum"],
        "rain_sum": daily["rain_sum"],
        "wind_speed": daily["wind_speed_10m_max"]
    })

    return df


# -----------------------------
# Streamlit UI
# -----------------------------

st.title("Прогноз опадів (ML)")

st.header("1. Отримання даних")

latitude = st.number_input("Latitude", value=50.45)
longitude = st.number_input("Longitude", value=30.52)

start_date = st.date_input("Дата початку")
end_date = st.date_input("Дата кінця")

if st.button("Отримати дані з Open-Meteo"):

    df = get_weather_data(latitude, longitude, start_date, end_date)

    df.to_csv("weather_daily.csv", index=False)

    st.success("Дані завантажені і збережені у weather_daily.csv")

    st.dataframe(df)


# -----------------------------
# Завантаження CSV
# -----------------------------

st.header("2. Завантаження CSV")

uploaded_file = st.file_uploader("Завантажте CSV", type=["csv"])

if uploaded_file:

    data = pd.read_csv(uploaded_file)

    st.write("Дані:")
    st.dataframe(data)


    # -----------------------------
    # Підготовка даних
    # -----------------------------

    data["label"] = (data["precipitation_sum"] > 0).astype(int)

    features = [
        "temp_max",
        "temp_min",
        "rain_sum",
        "wind_speed"
    ]

    X = data[features]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


    # -----------------------------
    # Навчання моделі
    # -----------------------------

    if st.button("Навчити модель"):

        model = LogisticRegression()

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        acc = accuracy_score(y_test, predictions)

        st.write("Accuracy:", acc)

        st.text("Classification report:")
        st.text(classification_report(y_test, predictions))

        st.session_state.model = model
        st.session_state.data = data


# -----------------------------
# Прогноз
# -----------------------------

st.header("3. Прогноз опадів")

if "model" in st.session_state:

    data = st.session_state.data
    model = st.session_state.model

    index = st.slider(
        "Оберіть день із датасету",
        0,
        len(data)-1
    )

    row = data.iloc[index]

    features = [
        row["temp_max"],
        row["temp_min"],
        row["rain_sum"],
        row["wind_speed"]
    ]

    if st.button("Зробити прогноз"):

        prediction = model.predict([features])[0]
        probability = model.predict_proba([features])[0][1]

        if prediction == 1:
            st.success(f"Очікуються опади. Ймовірність: {probability:.2f}")
        else:
            st.info(f"Опадів не очікується. Ймовірність опадів: {probability:.2f}")

