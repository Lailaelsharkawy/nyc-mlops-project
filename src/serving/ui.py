import streamlit as st
import requests

st.set_page_config(page_title="NYC Taxi Fare Predictor", page_icon="🚖")
st.title("🚖 NYC Taxi Fare Predictor")


hour_display = {
    0: "12:00 AM (Midnight)", 1: "1:00 AM", 2: "2:00 AM", 3: "3:00 AM", 
    4: "4:00 AM", 5: "5:00 AM", 6: "6:00 AM", 7: "7:00 AM", 8: "8:00 AM", 
    9: "9:00 AM", 10: "10:00 AM", 11: "11:00 AM", 12: "12:00 PM (Noon)", 
    13: "1:00 PM", 14: "2:00 PM", 15: "3:00 PM", 16: "4:00 PM", 17: "5:00 PM", 
    18: "6:00 PM", 19: "7:00 PM", 20: "8:00 PM", 21: "9:00 PM", 22: "10:00 PM", 
    23: "11:00 PM"
}

col1, col2 = st.columns(2)

with col1:
    passenger_count = st.slider("Passenger Count", 1, 6, 1)
    trip_distance = st.number_input("Trip Distance (miles)", min_value=0.1, max_value=100.0, value=1.0)

with col2:
    selected_hour_label = st.selectbox("Pickup Time", options=list(hour_display.values()))
    pickup_hour = list(hour_display.keys())[list(hour_display.values()).index(selected_hour_label)]

if st.button("Calculate Fare", use_container_width=True):
    payload = {
        "passenger_count": int(passenger_count),
        "trip_distance": float(trip_distance),
        "pickup_hour": int(pickup_hour)
    }
    
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        if response.status_code == 200:
            fare = response.json().get("fare_amount")
            st.metric(label="Estimated Total Fare", value=f"${fare:.2f}")
            st.balloons()
        else:
            st.error(f"API Error: {response.text}")
    except Exception as e:
        st.error(f"Connection Error: {e}")