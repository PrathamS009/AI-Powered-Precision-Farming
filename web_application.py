import streamlit as st
import tensorflow as tf
import joblib
import importlib
import numpy as np
import pandas as pd
from PIL import Image
import os

# --------------------------- PAGE CONFIG ---------------------------
st.set_page_config(page_title="AI Powered Precision Farming", layout="centered")
st.title("🌾 AI Powered Precision Farming")
st.markdown("---")

# --------------------------- SAFE MODEL LOADER ---------------------------

# @st.cache(allow_output_mutation=True)
# def load_tf_model_safe(model_path: str):
#     """
#     Load a Keras model. If loading fails due to unknown custom ViT layers,
#     try to import likely symbols from vit_keras and retry with only the
#     symbols that actually exist in the installed vit_keras package.
#     """
#     from tensorflow.keras.models import load_model

#     try:
#         return load_model(model_path)
#     except ValueError as e:
#         msg = str(e)
#         # Quick check: if error doesn't look like a custom-layer problem, re-raise
#         if not any(token in msg for token in ("Custom>", "Unknown layer", "ClassToken", "AddPositionEmbs")):
#             raise

#     # Candidate names that some vit-keras versions use
#     candidate_names = [
#         "ClassToken", "AddPositionEmbs", "TransformerBlock", "MLPBlock",
#         "PatchEncoder", "Patches", "PositionEmbedding", "PatchesEncoder",
#         "MultiHeadSelfAttention", "PatchEmbedding"
#     ]

#     custom_objects = {}

#     # Try import variants and collect available attributes
#     vit_modules = []
#     try:
#         vit_pkg = importlib.import_module("vit_keras")
#         vit_modules.append(vit_pkg)
#     except Exception:
#         vit_pkg = None

#     # common submodules
#     for mod_name in ("vit_keras.vit", "vit_keras.layers", "vit_keras.models", "vit_keras.preprocessing"):
#         try:
#             vit_modules.append(importlib.import_module(mod_name))
#         except Exception:
#             pass

#     # Probe for candidate symbols
#     for mod in vit_modules:
#         for name in candidate_names:
#             if name not in custom_objects and hasattr(mod, name):
#                 custom_objects[name] = getattr(mod, name)

#     # If we still found nothing, try to import top-level names that some forks use
#     # (this is a last-ditch effort; we won't silently swallow real errors)
#     if not custom_objects:
#         # Try to import the package that provides some ViT layers if installed under different name
#         for alt_pkg in ("vit", "vit_keras.tf", "keras_vit", "keras_vit.vit"):
#             try:
#                 alt = importlib.import_module(alt_pkg)
#                 for name in candidate_names:
#                     if hasattr(alt, name):
#                         custom_objects[name] = getattr(alt, name)
#             except Exception:
#                 pass

#     if not custom_objects:
#         # Explicit, actionable error for the user
#         raise ValueError(
#             "Model failed to load because it uses custom ViT layers and no compatible "
#             "symbols were found in the installed vit_keras (or compatible) package. "
#             "Install a compatible vit-keras or pass the required custom_objects yourself. "
#             "E.g. run: pip install vit-keras"
#         )

#     # Retry loading with the discovered custom_objects
#     return load_model(model_path, custom_objects=custom_objects)


# --------------------------- SECTION 1: Crop Disease Detection ---------------------------
with st.expander("🌱 Crop Disease Detection"):
    st.subheader("Upload Crop Image for Disease Detection")

    crop_type = st.selectbox(
        "Select Crop Type",
        ["Select", "Brinjal", "Cauliflower", "Rice", "Maize"]
    )

    model_paths = {
        "Brinjal": r"Crop_Disease_Detection/BrinjalLeaf/brinjal_model.h5", 
        "Cauliflower": r"Crop_Disease_Detection/CauliflowerLeaf/final_cauliflower_model.h5",
        "Rice": r"Crop_Disease_Detection/RiceLeaf/RiceLeafDiseasePreTrainedModel.keras",
        "Maize": r"Crop_Disease_Detection/MaizeLeaf/final_maize_vit_model.h5"
    }

    label_maps = {
        "Brinjal": {0: "Diseased", 1: "Fresh"},
        "Cauliflower": {
            0: "Bacterial Spot Rot",
            1: "Black Rot",
            2: "Downy Mildew",
            3: "Healthy Cauli",
            4: "Healthy Leaf",
        },
        "Rice": {
            0: "Brown Spot",
            1: "Leaf Blast",
            2: "Bacterial Leaf Blight",
            3: "Healthy Leaf"
        },
        "Maize": {
            0: "Blight",
            1: "Common Rust",
            2: "Gray Leaf Spot",
            3: "Healthy Leaf",
            4: "Phosphorus Deficiency",
        }
    }

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if crop_type != "Select" and uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        if st.button("Analyze Disease", key="disease_analyze"):
            st.write("Processing...")

            model_path = model_paths[crop_type]

            # ---- Load model with correct custom layers if ViT ----
            try:
                if crop_type in ["Maize", "Cauliflower"]:  # ViT models
                    from vit_keras import layers as vit_layers
                    custom_objects = {
                        "ClassToken": vit_layers.ClassToken,
                        "AddPositionEmbs": vit_layers.AddPositionEmbs,
                        "TransformerBlock": vit_layers.TransformerBlock,
                    }
                    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
                else:
                    model = tf.keras.models.load_model(model_path)
            except Exception as e:
                st.error(f"Model load failed: {e}")
                st.stop()

            # ---- Preprocessing logic per crop type ----
            img = Image.open(uploaded_image).convert("RGB").resize((224, 224))
            img_array = np.array(img).astype("float32")

            if crop_type in ["Maize", "Cauliflower", "Rice"]:  # ViT → 0–1 scaling
                img_array = img_array / 255.0
            elif crop_type == "Brinjal":
                # Brinjal binary classifier: sigmoid output
                img_array = img_array / 255.0

            img_array = np.expand_dims(img_array, axis=0)

            # ---- Prediction logic ----
            predictions = model.predict(img_array)

            if crop_type == "Brinjal":
                # sigmoid output: shape (1,1)
                prob = float(predictions[0][0])
                predicted_class = 1 if prob > 0.5 else 0
                confidence = prob if prob > 0.5 else 1 - prob
                class_name = label_maps[crop_type][predicted_class]
                st.success(f"Predicted: {class_name} (confidence {confidence:.2f})")

            else:
                # softmax output
                predicted_class = int(np.argmax(predictions, axis=1)[0])
                confidence = float(np.max(predictions))
                class_name = label_maps[crop_type].get(predicted_class, f"Class {predicted_class}")
                st.success(f"Predicted: {class_name} (confidence {confidence:.2f})")

        if st.button("Clear Image", key="disease_clear"):
            st.experimental_rerun()


# --------------------------- SECTION 2: Yield Prediction ---------------------------
with st.expander("🌾 Yield Prediction"):
    st.subheader("Enter Crop and Field Parameters")

    soil_type = st.selectbox("Soil Type", ["Sandy", "Loam", "Chalky", "Silt", "Clay"])
    crop_type_yield = st.selectbox("Crop Type", ["Maize", "Rice", "Wheat"])
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, key="yield_rainfall")
    temp = st.number_input("Temperature (°C)", min_value=-10.0, key="yield_temp")
    fertilizer_used = st.selectbox("Fertilizer Used", ["True", "False"], key="yield_fert")
    irrigation_used = st.selectbox("Irrigation Used", ["True", "False"], key="yield_irrig")
    weather = st.selectbox("Weather Condition", ["Sunny", "Rainy", "Cloudy"], key="yield_weather")
    days_to_harvest = st.number_input("Days to Harvest", min_value=0.0, key="yield_days")

    if st.button("Analyze Yield", key="yield_analyze"):
        st.write("Predicting yield...")

        yield_model_path = r"Yield_Prediction/yield_xgb_model.pkl"
        encoder_path = r"Yield_Prediction/yield_encoders.pkl"

        yield_model = joblib.load(yield_model_path)
        encoders = joblib.load(encoder_path)

        data = pd.DataFrame([{
            "Soil_Type": soil_type,
            "Crop": crop_type_yield,
            "Rainfall_mm": rainfall,
            "Temperature_Celsius": temp,
            "Fertilizer_Used": fertilizer_used,
            "Irrigation_Used": irrigation_used,
            "Weather_Condition": weather,
            "Days_to_Harvest": days_to_harvest
        }])

        # Encode categorical fields
        for col in encoders:
            data[col] = encoders[col].transform(data[col])

        # Convert boolean-like text columns to real bools or ints
        for col in ["Fertilizer_Used", "Irrigation_Used"]:
            if col in data.columns:
                data[col] = data[col].astype(str).str.lower().map({"true": 1, "false": 0}).astype(float)

        # Ensure all columns numeric
        for c in data.columns:
            if data[c].dtype == "object":
                try:
                    data[c] = pd.to_numeric(data[c])
                except Exception:
                    pass

        # st.write("Data types before predict:", data.dtypes)

        prediction = yield_model.predict(data)[0]
        st.success(f"Predicted Yield: {prediction:.2f} tons per hectare")


# --------------------------- SECTION 3: Irrigation Scheduling ---------------------------
with st.expander("💧 Irrigation Scheduling"):
    st.subheader("Input Environmental Data")

    temp_irrig = st.number_input("Temperature (°C)", min_value=-10.0, key="irr_temp")
    pressure_irrig = st.number_input("Pressure (hPa)", min_value=0.0, key="irr_pressure")
    altitude_irrig = st.number_input("Altitude (m)", min_value=-500.0, key="irr_altitude")
    soil_moisture = st.number_input("Soil Moisture (sensor value)", min_value=0.0, key="irr_moisture")

    if st.button("Analyze Irrigation", key="irr_analyze"):
        st.write("Analyzing irrigation schedule...")

        # Paths
        irr_model_path = r"Irrigation_Scheduling/irrigation_xgb_model.pkl"
        irr_encoder_path = r"Irrigation_Scheduling/class_label_encoder.pkl"
        irr_scaler_path = r"Irrigation_Scheduling/feature_scaler.pkl"

        # Load model, encoder, and scaler
        irr_model = joblib.load(irr_model_path)
        irr_encoder = joblib.load(irr_encoder_path)
        irr_scaler = joblib.load(irr_scaler_path)

        # Prepare input
        input_data = np.array([[temp_irrig, pressure_irrig, altitude_irrig, soil_moisture]])
        input_scaled = irr_scaler.transform(input_data)

        # Predict
        pred_class_idx = irr_model.predict(input_scaled)[0]
        predicted_class = irr_encoder.inverse_transform([pred_class_idx])[0]

        st.success(f"💧 Predicted Irrigation Class: {predicted_class}")

    if st.button("Clear Inputs", key="irr_clear"):
        st.experimental_rerun()



# --------------------------- FOOTER ---------------------------
st.markdown("---")
st.caption("Developed with ❤️ by Precision Farming AI Team")
