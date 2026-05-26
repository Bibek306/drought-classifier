import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

@st.cache_resource
def load_models():
    rf    = joblib.load('rf_model.pkl')
    sc    = joblib.load('scaler.pkl')
    nn    = tf.keras.models.load_model('nn_model.h5')
    return rf, sc, nn

rf_model, scaler, nn_model = load_models()

labels = ['No Drought', 'D1 Moderate', 'D2 Severe', 'D3 Extreme', 'D4 Exceptional']

st.set_page_config(page_title='Drought Classification System', layout='wide')
st.title('Drought Classification System')
st.write('Enter weather readings using the sliders to predict drought level.')

st.sidebar.header('Weather Inputs')

prectot   = st.sidebar.slider('Precipitation (mm)',      0.0,  50.0,  5.0)
ps        = st.sidebar.slider('Surface Pressure (kPa)',  80.0, 110.0, 100.0)
qv2m      = st.sidebar.slider('Humidity',                0.0,  30.0,  10.0)
t2m       = st.sidebar.slider('Temperature (°C)',       -10.0, 50.0,  25.0)
t2mdew    = st.sidebar.slider('Dew Point (°C)',         -10.0, 40.0,  15.0)
t2mwet    = st.sidebar.slider('Wet Bulb Temp (°C)',     -10.0, 40.0,  15.0)
t2m_max   = st.sidebar.slider('Max Temperature (°C)',   -10.0, 55.0,  30.0)
t2m_min   = st.sidebar.slider('Min Temperature (°C)',   -10.0, 45.0,  20.0)
t2m_range = st.sidebar.slider('Temp Range (°C)',          0.0, 30.0,  10.0)
ts        = st.sidebar.slider('Earth Skin Temp (°C)',   -10.0, 55.0,  25.0)
ws10m     = st.sidebar.slider('Wind Speed 10m (m/s)',     0.0, 20.0,   5.0)
ws10m_max = st.sidebar.slider('Max Wind 10m (m/s)',       0.0, 30.0,   8.0)
ws10m_min = st.sidebar.slider('Min Wind 10m (m/s)',       0.0, 15.0,   2.0)
ws10m_rng = st.sidebar.slider('Wind Range 10m (m/s)',     0.0, 20.0,   6.0)
ws50m     = st.sidebar.slider('Wind Speed 50m (m/s)',     0.0, 30.0,   8.0)
ws50m_max = st.sidebar.slider('Max Wind 50m (m/s)',       0.0, 40.0,  12.0)
ws50m_min = st.sidebar.slider('Min Wind 50m (m/s)',       0.0, 20.0,   4.0)
ws50m_rng = st.sidebar.slider('Wind Range 50m (m/s)',     0.0, 25.0,   8.0)

input_data   = np.array([[prectot, ps, qv2m, t2m, t2mdew, t2mwet,
                           t2m_max, t2m_min, t2m_range, ts,
                           ws10m, ws10m_max, ws10m_min, ws10m_rng,
                           ws50m, ws50m_max, ws50m_min, ws50m_rng]])
scaled_input = scaler.transform(input_data)

if st.button('Predict Drought Level'):

    nn_probs  = nn_model.predict(scaled_input)[0]
    nn_pred   = int(np.argmax(nn_probs))
    nn_conf   = float(nn_probs[nn_pred]) * 100

    rf_pred   = int(rf_model.predict(scaled_input)[0])

    st.markdown('---')
    st.subheader('Primary Prediction (Neural Network)')
    st.markdown(f'### :{"red" if nn_pred >= 3 else "orange" if nn_pred == 2 else "green"}[{labels[nn_pred]}]')
    st.write(f'Confidence: **{nn_conf:.1f}%**')

    st.subheader('Confidence Breakdown')
    for i, label in enumerate(labels):
        st.progress(float(nn_probs[i]), text=f'{label}: {nn_probs[i]*100:.1f}%')

    st.markdown('---')
    st.subheader('Cross Check (Random Forest)')
    if rf_pred == nn_pred:
        st.success(f'Both models agree — **{labels[rf_pred]}**')
    else:
        st.warning(f'Models disagree — Neural Network says **{labels[nn_pred]}**, Random Forest says **{labels[rf_pred]}**. Treat prediction with caution.')

    st.markdown('---')
    st.subheader('What this means')
    advice = {
        0: 'No drought detected. Water availability is normal.',
        1: 'Moderate drought. Monitor water levels closely.',
        2: 'Severe drought. Consider water conservation measures.',
        3: 'Extreme drought. Restrict non essential water usage immediately.',
        4: 'Exceptional drought. Emergency water management required.'
    }
    st.info(advice[nn_pred])