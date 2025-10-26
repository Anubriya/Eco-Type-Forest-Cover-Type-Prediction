import streamlit as st
import pandas as pd
import joblib

# Load the trained Random Forest model
tuned_rf = joblib.load("forest_cover_tuned_rf.pkl")

st.title("Forest Cover Type Prediction")
st.write("""
This app predicts the **forest cover type** based on geographic and soil features.
Please enter values in the fields below. Guidance is provided for each input.
""")

# ---------- Numeric inputs ----------
st.header("Numeric Features")
elevation = st.number_input("Elevation (meters) - e.g., 1870 to 3850", min_value=1863, max_value=3849, value=2874)
aspect = st.number_input("Aspect (degrees, 0-360) - slope direction", min_value=0, max_value=360, value=141)
slope = st.number_input("Slope (degrees) - e.g., 0 to 61", min_value=0, max_value=61, value=12)
horiz_dist_hydro = st.number_input("Horizontal Distance to Hydrology (m)", min_value=0, max_value=1343, value=252)
vert_dist_hydro = st.number_input("Vertical Distance to Hydrology (m)", min_value=-146, max_value=554, value=35)
horiz_dist_road = st.number_input("Horizontal Distance to Roadways (m)", min_value=0, max_value=7117, value=3314)
hillshade_9am = st.number_input("Hillshade at 9am (0-254)", min_value=0, max_value=254, value=217)
hillshade_noon = st.number_input("Hillshade at Noon (0-254)", min_value=0, max_value=254, value=225)
hillshade_3pm = st.number_input("Hillshade at 3pm (0-254)", min_value=0, max_value=254, value=140)
horiz_dist_fire = st.number_input("Horizontal Distance to Fire Points (m)", min_value=0, max_value=7173, value=3045)
abs_vert_dist_hydro = st.number_input("Absolute Vertical Distance to Hydrology", min_value=0, max_value=554, value=37)
hillshade_range = st.number_input("Hillshade Range", min_value=-74, max_value=247, value=85)
hillshade_avg = st.number_input("Hillshade Average", min_value=102, max_value=214, value=194)

# ---------- Wilderness Area (binary) ----------
st.header("Wilderness Areas")
wilderness_options = ["Wilderness_Area_1", "Wilderness_Area_2", "Wilderness_Area_3", "Wilderness_Area_4"]
wilderness_values = []
for w in wilderness_options:
    val = st.selectbox(f"{w} (0=No, 1=Yes)", [0, 1], index=1 if w == "Wilderness_Area_1" else 0)
    wilderness_values.append(val)

# ---------- Soil Type (binary) ----------
st.header("Soil Types")
soil_values = [0]*40
soil_choice = st.selectbox("Select Soil Type (1-40)", [i for i in range(1, 41)])
soil_values[soil_choice-1] = 1

# Combine all features into a list
input_list = [
    elevation, aspect, slope, horiz_dist_hydro, vert_dist_hydro,
    horiz_dist_road, hillshade_9am, hillshade_noon, hillshade_3pm,
    horiz_dist_fire, *wilderness_values, *soil_values,
    abs_vert_dist_hydro, hillshade_range, hillshade_avg
]

# ---------- Use model's feature names if available ----------
if hasattr(tuned_rf, "feature_names_in_"):
    feature_names = tuned_rf.feature_names_in_
else:
    # fallback (ensure correct order)
    feature_names = [
        "Elevation","Aspect","Slope","Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology","Horizontal_Distance_To_Roadways",
        "Hillshade_9am","Hillshade_Noon","Hillshade_3pm","Horizontal_Distance_To_Fire_Points"
    ] + wilderness_options + [f"Soil_Type_{i}" for i in range(1,41)] + ["Abs_Vertical_Distance_To_Hydrology","Hillshade_Range","Hillshade_Avg"]

# Check feature length
if len(input_list) != len(feature_names):
    st.error(f"Feature length mismatch: Model expects {len(feature_names)}, but got {len(input_list)}")
    st.stop()

# Convert to DataFrame
input_df = pd.DataFrame([input_list], columns=feature_names)

# ---------- Prediction ----------
if st.button("Predict Forest Cover Type"):
    pred = tuned_rf.predict(input_df)[0]  # model returns string label
    st.success(f"Predicted Forest Cover Type: **{pred}**")
