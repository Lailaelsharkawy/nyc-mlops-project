import streamlit as st
import requests

st.set_page_config(page_title="NYC Taxi Fare Predictor", page_icon="🚖")
st.title("🚖 NYC Taxi Fare Predictor")

hour_display = {i: f"{i:02d}:00" for i in range(24)}
day_display = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}

col1, col2 = st.columns(2)

with col1:
    passenger_count = st.slider("Passenger Count", 1, 6, 1)
    trip_distance = st.number_input("Trip Distance (miles)", min_value=0.1, max_value=100.0, value=1.0)
    selected_day_label = st.selectbox("Day of Week", options=list(day_display.values()))
    day_of_week = list(day_display.keys())[list(day_display.values()).index(selected_day_label)]

with col2:
    selected_hour_label = st.selectbox("Pickup Time", options=list(hour_display.values()))
    pickup_hour = list(hour_display.keys())[list(hour_display.values()).index(selected_hour_label)]
    
    st.caption("Trip Coordinates (Defaulting to Manhattan)")
    pickup_lon = st.number_input("Pickup Longitude", value=-73.98, format="%.4f")
    pickup_lat = st.number_input("Pickup Latitude", value=40.75, format="%.4f")
    dropoff_lon = st.number_input("Dropoff Longitude", value=-73.99, format="%.4f")
    dropoff_lat = st.number_input("Dropoff Latitude", value=40.73, format="%.4f")

if st.button("Calculate Fare", use_container_width=True):
    payload = {
        "trip_distance": float(trip_distance),
        "passenger_count": int(passenger_count),
        "pickup_longitude": float(pickup_lon),
        "pickup_latitude": float(pickup_lat),
        "dropoff_longitude": float(dropoff_lon),
        "dropoff_latitude": float(dropoff_lat),
        "pickup_hour": int(pickup_hour),
        "day_of_week": int(day_of_week)
    }
    
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        if response.status_code == 200:
            fare = response.json().get("fare_amount")
            st.metric(label="Estimated Total Fare", value=f"${fare:.2f}")
            st.balloons()
        else:
            st.error(f"API Error: {response.json().get('detail')}")
    except Exception as e:
        st.error(f"Connection Error: {e}")