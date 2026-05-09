import streamlit as st
import tensorflow as tf
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import os
import requests
from datetime import datetime
import altair as alt

# --------------------------- PAGE CONFIG ---------------------------
st.set_page_config(
    page_title="AI Precision Farming",
    layout="wide",
    page_icon="🌾",
    initial_sidebar_state="collapsed"
)

# --------------------------- MODERN CSS STYLING ---------------------------
st.markdown("""
<style>
/* ===== GOOGLE FONTS ===== */
@import url('[fonts.googleapis.com](https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap)');
@import url('[fonts.googleapis.com](https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap)');

/* ===== ROOT VARIABLES ===== */
:root {
    --primary: #10b981;
    --primary-dark: #059669;
    --primary-light: #d1fae5;
    --secondary: #0ea5e9;
    --accent: #f59e0b;
    --bg-main: #f8fafc;
    --bg-card: #ffffff;
    --bg-gradient: linear-gradient(135deg, #ecfdf5 0%, #f0f9ff 50%, #fefce8 100%);
    --text-primary: #0f172a;
    --text-secondary: #64748b;
    --text-muted: #94a3b8;
    --border: #e2e8f0;
    --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
    --shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
    --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
    --shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1);
    --radius: 16px;
    --radius-lg: 24px;
    --radius-xl: 32px;
}

/* ===== GLOBAL RESET ===== */
*, *::before, *::after {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

/* ===== MAIN APP STYLES ===== */
.stApp {
    background: var(--bg-gradient);
    font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    min-height: 100vh;
}

/* ===== FINAL LIGHT THEME SELECTBOX ===== */

/* Main Selectbox Container */
[data-baseweb="select"] {
    background: #ffffff !important;

    border-radius: 16px !important;
}

.weather-header-title{
        color:white;

        font-size:2.5rem;

        font-weight:800;

        margin-bottom:0.4rem;
    }    
            
.lowertitle-header-title{
        color:white;

        font-size:1.1rem;

        font-weight:400;

        margin-bottom:0.4rem;
    }    

/* Input Area */
[data-baseweb="select"] > div {
    background: #ffffff !important;

    border: 2px solid #dbe4ee !important;

    border-radius: 16px !important;

    min-height: 70px !important;

    box-shadow: 0 4px 12px rgba(0,0,0,0.04) !important;

    transition: all 0.3s ease !important;
}

/* Hover */
[data-baseweb="select"] > div:hover {
    border-color: #10b981 !important;

    box-shadow: 0 8px 20px rgba(16,185,129,0.12) !important;
}

/* Focus */
[data-baseweb="select"] > div:focus-within {
    border-color: #10b981 !important;

    box-shadow: 0 0 0 4px rgba(16,185,129,0.12) !important;
}

/* Selected Text */
[data-baseweb="select"] span {
    color: #334155 !important;

    font-weight: 500 !important;

    font-size: 1rem !important;
}

/* Placeholder Text */
[data-baseweb="select"] input {
    color: #334155 !important;
}

/* Dropdown Popup */
div[data-baseweb="popover"] {
    background: #ffffff !important;

    border-radius: 16px !important;

    border: 1px solid #e2e8f0 !important;

    box-shadow: 0 16px 40px rgba(0,0,0,0.10) !important;

    overflow: hidden !important;
}

/* Dropdown List */
ul {
    background: #ffffff !important;
}

/* Dropdown Items */
li[role="option"] {
    background: #ffffff !important;

    color: #334155 !important;

    font-weight: 500 !important;

    padding: 14px 16px !important;

    transition: all 0.2s ease !important;
}

/* Hovered Item */
li[role="option"]:hover {
    background: #ecfdf5 !important;

    color: #059669 !important;
}

/* Selected Item */
li[aria-selected="true"] {
    background: #d1fae5 !important;

    color: #065f46 !important;

    font-weight: 600 !important;
}


/* Hide Streamlit branding */
#MainMenu, footer {visibility: hidden;}
.stDeployButton {display: none;}

/* ===== SIDEBAR STYLING ===== */
[data-testid="stSidebar"] {
    background: linear-gradient(1deg, #0f172a 0%, #1e293b 100%);
    border-right: none;
}

[data-testid="stSidebar"] > div:first-child {
    padding: 2rem 1.5rem;
}

[data-testid="stSidebar"] .stRadio > label {
    color: #e2e8f0 !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 1rem !important;
}

[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] {
    gap: 0.5rem;
}

[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;

    width: 100% !important;
    min-height: 60px !important;

    display: flex !important;
    align-items: center !important;

    padding: 1rem 1.25rem !important;
    margin: 0 !important;

    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    cursor: pointer;
}

[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:hover {
    background: rgba(16, 185, 129, 0.15);
    border-color: rgba(16, 185, 129, 0.3);
    transform: translateX(4px);
}

[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label[data-checked="true"],
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label[data-baseweb="radio"]:has(input:checked) {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    border-color: transparent;
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);
}

[data-testid="stSidebar"] .stRadio label span {
    color: #000 !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
}


/* ===== IRRIGATION RESULT TEXT FIX ===== */

.irrigation-result {
    color: #0f172a !important;
}

.irrigation-result * {
    color: #0f172a !important;
}

.irrigation-result [data-testid="stMetricValue"] {
    color: #059669 !important;
    font-weight: 700 !important;
}

.irrigation-result [data-testid="stMetricLabel"] {
    color: #334155 !important;
}
Smar
.irrigation-result p,
.irrigation-result li,
.irrigation-result span,
.irrigation-result div,
.irrigation-result h1,
.irrigation-result h2,
.irrigation-result h3,
.irrigation-result h4 {
    color: #0f172a !important;
}
            

            /* ===== ANALYSIS RESULT PLACEHOLDER ===== */

.analysis-placeholder {
    background: #ffffff;

    border: 2px dashed #dbe4ee;

    border-radius: 18px;

    padding: 3rem 2rem;

    text-align: center;

    transition: all 0.3s ease;

    box-shadow: 0 4px 12px rgba(0,0,0,0.04);
}

/* Hover Effect SAME as Selectbox */
.analysis-placeholder:hover {
    border-color: #10b981;

    background: #f8fafc;

    box-shadow:
        0 0 0 4px rgba(16,185,129,0.12),
        0 8px 24px rgba(16,185,129,0.14);

    transform: translateY(-2px);
}

/* ===== HERO HEADER ===== */
.hero-container {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #134e4a 100%);
    border-radius: var(--radius-xl);
    padding: 1.5rem 1.5rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow-xl);
}

.hero-container::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 60%;
    height: 200%;
    background: radial-gradient(circle, rgba(16, 185, 129, 0.15) 0%, transparent 60%);
    pointer-events: none;
}

.hero-container::after {
    content: '';
    position: absolute;
    bottom: -30%;
    left: -10%;
    width: 40%;
    height: 150%;
    background: radial-gradient(circle, rgba(14, 165, 233, 0.1) 0%, transparent 50%);
    pointer-events: none;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(16, 185, 129, 0.2);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #6ee7b7;
    padding: 0.5rem 1rem;
    border-radius: 100px;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.1rem;
}

.hero-title {
    font-size: 3rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1.1;
    margin-bottom: 1rem;
    position: relative;
    z-index: 1;
}

.hero-title .gradient-text {
    background: linear-gradient(135deg, #6ee7b7 0%, #38bdf8 50%, #fbbf24 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-subtitle {
    font-size: 0.95rem;
    color: #94a3b8;

    max-width: 100%;

    white-space: nowrap;

    overflow: hidden;
    text-overflow: ellipsis;

    line-height: 1.5;

    position: relative;
    z-index: 1;
}

/* ===== SECTION CARDS ===== */
.section-card {
    background: var(--bg-card);
    border-radius: var(--radius-lg);
    padding: 2rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow);
    border: 1px solid var(--border);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.section-card:hover {
    box-shadow: var(--shadow-lg);
    transform: translateY(-2px);
}

.section-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.5rem;
}

.section-icon {
    width: 56px;
    height: 56px;
    border-radius: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.75rem;
}

.section-icon.green { background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); }
.section-icon.blue { background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%); }
.section-icon.amber { background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); }
.section-icon.purple { background: linear-gradient(135deg, #ede9fe 0%, #ddd6fe 100%); }

.section-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0;
}

.section-subtitle {
    font-size: 0.9rem;
    color: var(--text-secondary);
    margin: 0.25rem 0 0 0;
}

/* ===== FORM INPUTS ===== */
.stTextInput > div > div > input,
.stNumberInput input,
.stSelectbox > div > div {
    background: #f8fafc !important;
    border: 2px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 0.875rem 1.25rem !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    color: var(--text-primary) !important;
    transition: all 0.2s ease !important;
}

.stTextInput > div > div > input:focus,
.stNumberInput input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 4px rgba(16, 185, 129, 0.1) !important;
    outline: none !important;
}

.stTextInput > label,
.stNumberInput label,
.stSelectbox label {
    font-size: 0.875rem !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    margin-bottom: 0.5rem !important;
}

/* ===== BUTTONS ===== */
.stButton > button {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.875rem 2rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 14px rgba(16, 185, 129, 0.35) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(16, 185, 129, 0.45) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
}

/* ===== METRICS GRID ===== */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
    margin: 1.5rem 0;
}

.metric-card {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-lg);
    border-color: var(--primary-light);
}

.metric-icon {
    font-size: 2rem;
    margin-bottom: 0.75rem;
    display: block;
}

.metric-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #000000;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
}

.metric-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: #444444 !important;
    line-height: 1;
}

.metric-subvalue {
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-top: 0.25rem;
}

/* ===== IRRIGATION RESULT TEXT BLACK ===== */

.stMetric label,
.stMetric div,
.stMetric span,
.stMetric p,
.stMarkdown,
.stMarkdown p,
.stMarkdown li,
.stAlert,
.stAlert p,
.stSuccess,
.stWarning,
.stError,
.stInfo,
.result-card,
.result-title,
.result-value,
.result-subtitle {
    color: #000000 !important;
}

/* Recommended Action headings */
h1, h2, h3, h4, h5, h6 {
    color: #000000 !important;
}

/* ===== STATUS BADGES ===== */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    border-radius: 100px;
    font-size: 0.85rem;
    font-weight: 600;
}

.status-badge.success {
    background: #d1fae5;
    color: #065f46;
}

.status-badge.warning {
    background: #fef3c7;
    color: #92400e;
}

.status-badge.error {
    background: #fee2e2;
    color: #991b1b;
}

/* ===== RESULT CARD ===== */
.result-card {
    background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
    border: 2px solid #a7f3d0;
    border-radius: var(--radius-lg);
    padding: 2rem;
    text-align: center;
    margin: 1.5rem 0;
}

.result-card.warning {
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    border-color: #fcd34d;
}

.result-card.error {
    background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
    border-color: #f87171;
}

.result-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
}

.result-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
}

.result-value {
    font-size: 2.5rem;
    font-weight: 800;
    color: #059669;
    margin-bottom: 0.5rem;
}

.result-subtitle {
    font-size: 0.9rem;
    color: var(--text-secondary);
}

/* ===== WEATHER SPECIFIC ===== */
.weather-hero {
    background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
    border-radius: var(--radius-xl);
    padding: 2.5rem;
    color: white;
    text-align: center;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}

.weather-hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -25%;
    width: 50%;
    height: 150%;
    background: radial-gradient(circle, rgba(255, 255, 255, 0.1) 0%, transparent 50%);
}

.weather-city {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

.weather-temp {
    font-size: 4rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 0.5rem;
}

.weather-condition {
    font-size: 1.1rem;
    opacity: 0.9;
}

/* ===== CHART CONTAINER ===== */
.chart-container {
    background: var(--bg-card);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    box-shadow: var(--shadow);
    border: 1px solid var(--border);
}

/* ===== TABLE STYLES ===== */
.custom-table-container {
    background: var(--bg-card);
    border-radius: var(--radius);
    overflow: hidden;
    box-shadow: var(--shadow);
    border: 1px solid var(--border);
}

.table-header {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    padding: 1rem 1.5rem;
    border-bottom: 1px solid var(--border);
}




.custom-table {
    width: 100%;
    border-collapse: collapse;
}

.custom-table th {
    background: #f8fafc;
    padding: 1rem;
    text-align: left;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-bottom: 1px solid var(--border);
}

.custom-table td {
    padding: 1rem;
    font-size: 0.9rem;
    color: var(--text-primary);
    border-bottom: 1px solid #f1f5f9;
}

.custom-table tr:hover td {
    background: #f8fafc;
}
            

            
/* ===== MODERN LIGHT FILE UPLOADER ===== */

[data-testid="stFileUploader"] {
    background: transparent !important;
}

/* Main Upload Box */
[data-testid="stFileUploader"] section {
    background: #ffffff !important;

    border: 2px dashed #dbe4ee !important;

    border-radius: 18px !important;

    padding: 1.5rem !important;

    transition: all 0.3s ease !important;

    box-shadow: 0 4px 12px rgba(0,0,0,0.04) !important;
}

/* Hover */
[data-testid="stFileUploader"] section:hover {
    border-color: #10b981 !important;

    background: #f8fafc !important;

    box-shadow: 0 8px 24px rgba(16,185,129,0.12) !important;
}

/* Upload Text */
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p {
    color: #475569 !important;

    font-weight: 500 !important;
}

/* Browse Files Button */
[data-testid="stFileUploader"] button {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;

    color: white !important;

    border: none !important;

    border-radius: 12px !important;

    font-weight: 600 !important;

    padding: 0.6rem 1.2rem !important;

    transition: 0.3s ease !important;
}

/* Button Hover */
[data-testid="stFileUploader"] button:hover {
    transform: translateY(-1px);

    box-shadow: 0 6px 16px rgba(16,185,129,0.25) !important;
}

/* Remove dark inner background */
[data-testid="stFileUploaderDropzone"] {
    background: #ffffff !important;
}

/* Drag text area */
[data-testid="stFileUploaderDropzoneInstructions"] {
    color: #475569 !important;
}
/* ===== IMAGE DISPLAY ===== */
.stImage {
    border-radius: var(--radius);
    overflow: hidden;
    box-shadow: var(--shadow);
}

/* ===== TABS ===== */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: #f1f5f9;
    padding: 0.5rem;
    border-radius: 12px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-weight: 600;
    color: var(--text-secondary);
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: white;
    color: var(--text-primary);
    box-shadow: var(--shadow-sm);
}

/* ===== ALERTS ===== */
.stSuccess, .stError, .stWarning, .stInfo {
    border-radius: var(--radius) !important;
    border: none !important;
    padding: 1rem 1.25rem !important;
}

/* ===== FOOTER ===== */
.footer {
    text-align: center;
    padding: 2rem;
    color: var(--text-muted);
    font-size: 0.9rem;
}

.footer a {
    color: var(--primary);
    text-decoration: none;
    font-weight: 600;
}

/* ===== RESPONSIVE ===== */
@media (max-width: 768px) {
    .hero-title { font-size: 2rem; }
    .hero-container { padding: 0.5rem; }
    .metrics-grid { grid-template-columns: repeat(2, 1fr); }
    .section-card { padding: 1.5rem; }
}

/* ===== ANIMATIONS ===== */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.animate-in {
    animation: fadeInUp 0.5s ease-out forwards;
}

/* ===== LOADING SPINNER ===== */
.stSpinner > div {
    border-color: var(--primary) transparent transparent transparent !important;
}
</style>
""", unsafe_allow_html=True)

# --------------------------- SIDEBAR NAVIGATION ---------------------------
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1rem 0 2rem 0;'>
        <div style='
            width: 80px; 
            height: 80px; 
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            border-radius: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 1rem auto;
            font-size: 2.5rem;
            box-shadow: 0 8px 25px rgba(16, 185, 129, 0.35);
        '>🌾</div>
       
        
    </div>
    """, unsafe_allow_html=True)
    
    section = st.radio(
        "NAVIGATION",
        [
            "🌱 Crop Disease Detection",
            "🌾 Yield Prediction",
            "💧 Irrigation Scheduling",
            "🌤️ Weather Dashboard"
        ],
        index=0,
        label_visibility="visible"
    )
    
    st.markdown("""
    <div style='
        margin-top: 3rem;
        padding: 1.25rem;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-radius: 12px;
    '>
        <span class="lowertitle-header-title">            
            💡 Quick Tip
        </span>
                <p></p>
        <span class="lowertitle-header-title">
            Upload clear, well-lit images for best disease detection results.
        </span>
    </div>
    """, unsafe_allow_html=True)

# --------------------------- HERO HEADER ---------------------------
st.markdown("""
<div class="hero-container animate-in">
    <div class="hero-badge">
        <span>✨</span> AI-Powered Platform
    </div>
    <h1 class="hero-title">
       <span class="weather-header-title"> Smart <span class="gradient-text">Precision Farming</span><br>
        for Modern Agriculture</span>
    </h1>
    <span class="lowertitle-header-title">
        Harness the power of artificial intelligence for crop disease detection, yield prediction, intelligent irrigation scheduling, and real-time weather insights.
    </span>
</div>
""", unsafe_allow_html=True)

# --------------------------- MODEL PATHS & LABEL MAPS ---------------------------
model_paths = {
    "Brinjal": r"Crop_Disease_Detection/BrinjalLeaf/brinjal_model.h5",
    "Cauliflower": r"Crop_Disease_Detection/CauliflowerLeaf/final_cauliflower_model.h5",
    "Rice": r"Crop_Disease_Detection/RiceLeaf/RiceLeafDiseasePreTrainedModel.keras",
    "Maize": r"Crop_Disease_Detection/MaizeLeaf/maize_vit_best.h5"
}

label_maps = {
    "Brinjal": {0: "Diseased", 1: "Fresh"},
    "Cauliflower": {
        0: "Bacterial Spot Rot",
        1: "Black Rot",
        2: "Downy Mildew",
        3: "Healthy Cauliflower",
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

# ========================== MAIN CONTENT ==========================

if section == "🌱 Crop Disease Detection":
    st.markdown("""
        <div class="section-header">
            <div class="section-icon green">🌱</div>
            <div>
            <p style='color: #5b5b5b; font-size: 2rem; margin: 0; line-height: 1.5;'>
                        Crop Disease Detection
                    </p>
                            </div>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("""
            <div>
            <p style='color: #5b5b5b; font-size: 1.5rem; margin: 0; line-height: 1.5;'>
                        📋 Configuration
                    </p>
                            </div>
        </div>
    """, unsafe_allow_html=True)
        
        crop_type = st.selectbox(
            "Select Crop Type",
            ["Select a crop...", "Brinjal", "Cauliflower", "Rice", "Maize"],
            help="Choose the type of crop for disease analysis"
            
            
        )
        
        
        uploaded_image = st.file_uploader(
            "Upload Leaf Image",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG"
        )
        
        if uploaded_image:
            st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        st.markdown("""
            <div>
            <p style='color: #5b5b5b; font-size: 1.5rem; margin-bottom: 0.5rem; line-height: 1.5;'>
                        🔬 Analysis Results
                    </p>
                            </div>
        </div>
    """, unsafe_allow_html=True)
        
        if crop_type != "Select a crop..." and uploaded_image is not None:
            if st.button("🔍 Analyze Disease", use_container_width=True):
                with st.spinner("Analyzing image with AI model..."):
                    model_path = model_paths[crop_type]
                    
                    try:
                        if crop_type in ["Maize", "Cauliflower"]:
                            from vit_keras import layers as vit_layers
                            custom_objects = {
                                "ClassToken": vit_layers.ClassToken,
                                "AddPositionEmbs": vit_layers.AddPositionEmbs,
                                "TransformerBlock": vit_layers.TransformerBlock,
                            }
                            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
                        else:
                            model = tf.keras.models.load_model(model_path)
                        
                        img = Image.open(uploaded_image).convert("RGB").resize((224, 224))
                        img_array = np.array(img).astype("float32") / 255.0
                        img_array = np.expand_dims(img_array, axis=0)
                        
                        predictions = model.predict(img_array)
                        
                        if crop_type == "Brinjal":
                            prob = float(predictions[0][0])
                            predicted_class = 1 if prob > 0.5 else 0
                            confidence = prob if prob > 0.5 else 1 - prob
                        else:
                            predicted_class = int(np.argmax(predictions, axis=1)[0])
                            confidence = float(np.max(predictions))
                        
                        class_name = label_maps[crop_type][predicted_class]
                        is_healthy = "Healthy" in class_name or "Fresh" in class_name
                        
                        result_class = "success" if is_healthy else "warning"
                        result_icon = "✅" if is_healthy else "⚠️"
                        
                        st.markdown(f"""
                        <div class="result-card {'' if is_healthy else 'warning'}">
                            <div class="result-icon">{result_icon}</div>
                            <div class="result-title">Detection Complete</div>
                            <div class="result-value" style="color: {'#059669' if is_healthy else '#d97706'};">
                                {class_name}
                            </div>
                            <div class="result-subtitle">Confidence: {confidence:.1%}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if not is_healthy:
                            st.info("💡 **Recommendation**: Consult with an agricultural expert for treatment options.")
                    
                    except Exception as e:
                        st.error(f"Error loading model: {e}")
        else:
            st.markdown("""
                        <div class="analysis-placeholder">
            <div style='
                background: #f8fafc;
                border-radius: 16px;
                padding: 0.5rem 0.5rem;
                text-align: center;
            '>
                <div style='font-size: 3rem; margin-bottom: 1rem;'>🔬</div>
                <p style='color: #64748b; font-size: 1rem; margin: 0;'>
                    Select a crop type and upload an image to begin analysis
                </p>
            </div>
                        </div>
            """, unsafe_allow_html=True)


elif section == "🌾 Yield Prediction":

    st.markdown("## 🌾 Crop Yield Prediction")

    col1, col2 = st.columns([1, 1], gap="large")

    # =========================================================
    # INPUT SECTION
    # =========================================================

    with col1:

        st.subheader("🧪 Soil & Environmental Parameters")

        fertilizer_amt = st.number_input(
            "Fertilizer Amount (kg/ha)",
            min_value=0.0,
            value=120.0,
            format="%.2f",
            key="yield_fertilizer",
        )

        temp_yield = st.number_input(
            "Temperature (°C)",
            min_value=-10.0,
            max_value=60.0,
            value=28.0,
            format="%.1f",
            key="yield_temp",
        )

        rainfall = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            value=180.0,
            format="%.1f",
            key="yield_rainfall",
        )

        humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=68.0,
            format="%.1f",
            key="yield_humidity",
        )

        st.subheader("🔬 NPK Values")

        n_col, p_col, k_col = st.columns(3)

        with n_col:
            N_val = st.number_input(
                "Nitrogen (N)",
                min_value=0.0,
                value=75.0,
                format="%.2f",
                key="yield_N"
            )

        with p_col:
            P_val = st.number_input(
                "Phosphorus (P)",
                min_value=0.0,
                value=42.0,
                format="%.2f",
                key="yield_P"
            )

        with k_col:
            K_val = st.number_input(
                "Potassium (K)",
                min_value=0.0,
                value=55.0,
                format="%.2f",
                key="yield_K"
            )

    # =========================================================
    # RESULT SECTION
    # =========================================================

    with col2:

        st.subheader("📊 Prediction Results")

        if st.button("📈 Predict Yield", use_container_width=True):

            with st.spinner("Analyzing agricultural parameters..."):

                score = 100
                reasons = []

                # =========================================================
                # TEMPERATURE ANALYSIS
                # =========================================================

                if temp_yield < 0:

                    score -= 70

                    reasons.append(
                        f"❌ Extreme freezing temperature ({temp_yield}°C) can severely damage crops"
                    )

                elif temp_yield < 10:

                    score -= 45

                    reasons.append(
                        f"⚠️ Very low temperature ({temp_yield}°C) slows crop growth significantly"
                    )

                elif 22 <= temp_yield <= 32:

                    reasons.append(
                        f"✅ Temperature ({temp_yield}°C) is optimal for crop growth"
                    )

                elif temp_yield > 42:

                    score -= 60

                    reasons.append(
                        f"🔥 Extreme heat ({temp_yield}°C) can destroy crop productivity"
                    )

                elif temp_yield > 36:

                    score -= 35

                    reasons.append(
                        f"⚠️ High temperature ({temp_yield}°C) may stress crops"
                    )

                else:

                    score -= 10

                    reasons.append(
                        f"⚠️ Temperature is not fully optimal"
                    )

                # =========================================================
                # FERTILIZER ANALYSIS
                # =========================================================

                if fertilizer_amt < 20:

                    score -= 35

                    reasons.append(
                        "❌ Fertilizer level is critically low"
                    )

                elif fertilizer_amt < 60:

                    score -= 15

                    reasons.append(
                        "⚠️ Fertilizer level is below recommended range"
                    )

                elif 80 <= fertilizer_amt <= 180:

                    reasons.append(
                        "✅ Fertilizer amount is balanced"
                    )

                elif fertilizer_amt > 300:

                    score -= 25

                    reasons.append(
                        "⚠️ Excess fertilizer may damage soil quality"
                    )

                # =========================================================
                # RAINFALL ANALYSIS
                # =========================================================

                if rainfall < 30:

                    score -= 45

                    reasons.append(
                        f"❌ Severe lack of rainfall ({rainfall} mm)"
                    )

                elif rainfall < 80:

                    score -= 20

                    reasons.append(
                        f"⚠️ Rainfall is below optimal range"
                    )

                elif 120 <= rainfall <= 250:

                    reasons.append(
                        f"✅ Rainfall is favorable"
                    )

                elif rainfall > 400:

                    score -= 35

                    reasons.append(
                        f"⚠️ Excess rainfall may cause flooding and root damage"
                    )

                # =========================================================
                # HUMIDITY ANALYSIS
                # =========================================================

                if humidity < 20:

                    score -= 30

                    reasons.append(
                        f"❌ Extremely low humidity ({humidity}%)"
                    )

                elif humidity > 95:

                    score -= 25

                    reasons.append(
                        f"⚠️ Excess humidity may promote fungal diseases"
                    )

                elif 50 <= humidity <= 75:

                    reasons.append(
                        f"✅ Humidity is suitable"
                    )

                else:

                    score -= 10

                    reasons.append(
                        f"⚠️ Humidity conditions are not ideal"
                    )

                # =========================================================
                # NPK ANALYSIS
                # =========================================================

                npk_avg = (N_val + P_val + K_val) / 3

                if npk_avg < 15:

                    score -= 40

                    reasons.append(
                        "❌ Soil nutrients are critically deficient"
                    )

                elif npk_avg < 35:

                    score -= 20

                    reasons.append(
                        "⚠️ Nutrient levels are low"
                    )

                elif 40 <= npk_avg <= 80:

                    reasons.append(
                        "✅ NPK nutrient balance is healthy"
                    )

                elif npk_avg > 120:

                    score -= 15

                    reasons.append(
                        "⚠️ Excess nutrients may reduce soil stability"
                    )

                # =========================================================
                # FINAL SCORE CLAMP
                # =========================================================

                score = max(0, min(score, 100))

                # =========================================================
                # YIELD CALCULATION
                # =========================================================

                predicted_yield = round((score / 100) * 6.5, 2)

                # =========================================================
                # FINAL CLASSIFICATION
                # =========================================================

                if predicted_yield >= 5:

                    st.success("🌟 Excellent Yield Expected")

                    category = "Excellent"

                elif predicted_yield >= 3.5:

                    st.info("✅ Good Yield Expected")

                    category = "Good"

                elif predicted_yield >= 2:

                    st.warning("⚠️ Average Yield Expected")

                    category = "Average"

                else:

                    st.error("❌ Low Yield Expected")

                    category = "Low"

                # =========================================================
                # METRICS
                # =========================================================

                metric_col1, metric_col2 = st.columns(2)

                with metric_col1:
                    st.metric(
                        label="Predicted Yield",
                        value=f"{predicted_yield} t/ha"
                    )

                with metric_col2:
                    st.metric(
                        label="AI Productivity Score",
                        value=f"{score}/100"
                    )

                st.subheader("📋 AI Analysis Summary")

                for reason in reasons:
                    st.write(f"- {reason}")

            
elif section == "💧 Irrigation Scheduling":
    
    st.markdown("## 💧 Smart Irrigation Scheduling")

    col1, col2 = st.columns([1, 1], gap="large")

    # =========================================================
    # LEFT SIDE INPUTS
    # =========================================================

    with col1:

        st.markdown("""
        <p style='
            color: #5b5b5b;
            font-size: 1.5rem;
            margin-bottom: 1rem;
            line-height: 1.5;
            font-weight: 600;
        '>
            🌡️ Environmental Inputs
        </p>
        """, unsafe_allow_html=True)

        temp_irrig = st.number_input(
            "Temperature (°C)",
            min_value=-10.0,
            value=25.0
        )

        pressure_irrig = st.number_input(
            "Atmospheric Pressure (hPa)",
            min_value=800.0,
            value=1013.25
        )

        altitude_irrig = st.number_input(
            "Altitude (meters)",
            value=100.0
        )

        soil_moisture = st.number_input(
            "Soil Moisture Sensor Value",
            min_value=0.0,
            max_value=100.0,
            value=45.0
        )

    # =========================================================
    # RIGHT SIDE RESULTS
    # =========================================================

    with col2:

        st.markdown("""
        <p style='
            color: #5b5b5b;
            font-size: 1.5rem;
            margin-bottom: 1rem;
            line-height: 1.5;
            font-weight: 600;
        '>
            🚿 Irrigation Recommendation
        </p>
        """, unsafe_allow_html=True)

        if st.button("💧 Get Recommendation", width="stretch"):

            with st.spinner("Analyzing conditions..."):

                # =================================================
                # MATHEMATICAL IRRIGATION INDEX
                # =================================================

                moisture_score = max(0, 100 - soil_moisture)

                temp_score = temp_irrig * 1.8

                pressure_score = max(
                    0,
                    1013 - pressure_irrig
                ) * 0.5

                altitude_score = altitude_irrig * 0.002

                irrigation_index = (
                    moisture_score * 0.5 +
                    temp_score * 0.3 +
                    pressure_score * 0.1 +
                    altitude_score * 0.1
                )

                # =================================================
                # CLASSIFICATION
                # =================================================

                if irrigation_index >= 65:
                    predicted_class = "High"

                elif irrigation_index >= 40:
                    predicted_class = "Medium"

                else:
                    predicted_class = "Low"

                # =================================================
                # REASON GENERATION
                # =================================================

                reasons = []

                # Soil Moisture
                if soil_moisture < 20:

                    reasons.append(
                        f"🔻 Soil moisture is critically low ({soil_moisture})"
                    )

                elif soil_moisture < 40:

                    reasons.append(
                        f"⚠️ Soil moisture is below optimal range ({soil_moisture})"
                    )

                else:

                    reasons.append(
                        f"✅ Soil moisture is adequate ({soil_moisture})"
                    )

                # Temperature
                if temp_irrig > 38:

                    reasons.append(
                        f"🔥 Temperature is extremely high ({temp_irrig}°C)"
                    )

                elif temp_irrig > 30:

                    reasons.append(
                        f"🌡️ Temperature is moderately high ({temp_irrig}°C)"
                    )

                elif temp_irrig < 10:

                    reasons.append(
                        f"❄️ Temperature is very low ({temp_irrig}°C)"
                    )

                else:

                    reasons.append(
                        f"✅ Temperature is within optimal range ({temp_irrig}°C)"
                    )

                # Pressure
                if pressure_irrig < 990:

                    reasons.append(
                        f"🌪️ Atmospheric pressure is unusually low ({pressure_irrig} hPa)"
                    )

                elif pressure_irrig > 1025:

                    reasons.append(
                        f"☀️ High atmospheric pressure detected ({pressure_irrig} hPa)"
                    )

                else:

                    reasons.append(
                        f"✅ Atmospheric pressure is stable ({pressure_irrig} hPa)"
                    )

                # Altitude
                if altitude_irrig > 1500:

                    reasons.append(
                        f"⛰️ High altitude may increase evaporation ({altitude_irrig} m)"
                    )

                elif altitude_irrig < 100:

                    reasons.append(
                        f"🌍 Low altitude conditions detected ({altitude_irrig} m)"
                    )

                # =================================================
                # RESULT WRAPPER
                # =================================================

                st.markdown(
                    '<div class="irrigation-result">',
                    unsafe_allow_html=True
                )

                # =================================================
                # METRIC
                # =================================================

                st.metric(
                    label="Irrigation Index",
                    value=f"{irrigation_index:.2f}"
                )

                # =================================================
                # MAIN RESULT
                # =================================================

                if predicted_class == "High":

                    st.error(
                        "🔴 High Irrigation Required"
                    )

                    st.markdown("""
                    ### Recommended Action

                    Immediate irrigation is strongly recommended
                    to prevent crop stress and moisture deficiency.
                    """)

                elif predicted_class == "Medium":

                    st.warning(
                        "🟡 Moderate Irrigation Recommended"
                    )

                    st.markdown("""
                    ### Recommended Action

                    Irrigation may be needed soon.
                    Continue monitoring environmental conditions.
                    """)

                else:

                    st.success(
                        "🟢 Low Irrigation Needed"
                    )

                    st.markdown("""
                    ### Recommended Action

                    Current environmental conditions are stable.
                    No immediate irrigation is required.
                    """)

                # =================================================
                # ANALYSIS SUMMARY
                # =================================================

                st.markdown("### 📋 Analysis Summary")

                for reason in reasons:

                    st.markdown(f"- {reason}")

                # =================================================
                # CLOSE WRAPPER
                # =================================================

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True
                )

                    
                
                    
        
elif section == "🌤️ Weather Dashboard":
    API_KEY = "fd9e52f26737bfdca543f08b13356d67"
    HISTORY_FILE = "weather_history.csv"
    
    # =========================================================
    # PREMIUM WEATHER CSS
    # =========================================================

    st.markdown("""
    <style>

    /* =========================
    MAIN WEATHER CONTAINER
    ========================= */

    .weather-wrapper{
        animation: fadeUp 0.7s ease;
    }

    /* =========================
    HEADER CARD
    ========================= */

    .weather-header{
        background:
            linear-gradient(
                135deg,
                rgba(10,25,60,0.95),
                rgba(5,15,40,0.98)
            );

        border:1px solid rgba(255,255,255,0.08);

        border-radius:24px;

        padding:2rem;

        margin-bottom:1.5rem;

        box-shadow:
            0 0 30px rgba(0,0,0,0.25);
    }

    .weather-header-title{
        color:white;

        font-size:2.5rem;

        font-weight:800;

        margin-bottom:0.4rem;
    }

    .weather-header-sub{
        color:#94a3b8;

        font-size:1rem;
    }

    /* =========================
    SEARCH CARD
    ========================= */

    .search-card{
        background:
            linear-gradient(
                135deg,
                rgba(10,25,60,0.95),
                rgba(5,15,40,0.98)
            );

        border:1px solid rgba(255,255,255,0.08);

        border-radius:24px;

        padding:1.8rem;

        margin-bottom:1.5rem;
    }

    /* =========================
    INPUTS
    ========================= */

    .stTextInput input{
        background:#071225 !important;

        border:1px solid rgba(255,255,255,0.08) !important;

        color:black !important;

        border-radius:14px !important;

        padding:0.9rem !important;
    }

    .stTextInput label{
        color:black !important;

        font-weight:600 !important;
    }

    .stSelectbox label{
        color:black !important;

        font-weight:600 !important;
    }

    .stSelectbox > div > div{
        background:#071225 !important;

        border:1px solid rgba(255,255,255,0.08) !important;

        border-radius:14px !important;

        color:white !important;
    }

    /* =========================
    BUTTONS
    ========================= */

    .stButton button{
        width:100%;

        background:
            linear-gradient(
                135deg,
                #10b981,
                #14b8a6
            ) !important;

        border:none !important;

        border-radius:14px !important;

        color:white !important;

        font-weight:700 !important;

        padding:0.9rem !important;

        transition:0.3s ease !important;
    }

    .stButton button:hover{
        transform: translateY(-3px);

        box-shadow:
            0 10px 25px rgba(16,185,129,0.35);
    }

    /* =========================
    MAIN WEATHER HERO
    ========================= */

    .main-weather-card{
        background:
            linear-gradient(
                135deg,
                rgba(10,25,60,0.98),
                rgba(5,15,40,0.98)
            );

        border:1px solid rgba(255,255,255,0.08);

        border-radius:26px;

        padding:2rem;

        margin-bottom:1.5rem;

        box-shadow:
            0 0 40px rgba(0,0,0,0.25);
    }

    .weather-location{
        color:white;

        font-size:3rem;

        font-weight:800;
    }

    .weather-country{
        color:#94a3b8;

        margin-top:0.3rem;
    }

    .weather-center{
        display:flex;

        align-items:center;

        justify-content:center;

        gap:2rem;

        margin-top:2rem;
    }

    .weather-icon{
        font-size:8rem;
    }

    .weather-temp{
        font-size:7rem;

        font-weight:900;

        color:white;

        line-height:1;
    }

    .weather-condition{
        color:#10b981;

        font-size:2rem;

        margin-top:0.5rem;

        font-weight:700;
    }

    /* =========================
    LIVE BADGE
    ========================= */

    .live-badge{
        background:
            linear-gradient(
                135deg,
                #10b981,
                #14b8a6
            );

        color:white;

        padding:0.4rem 1rem;

        border-radius:999px;

        font-size:0.8rem;

        font-weight:700;

        float:right;

        box-shadow:
            0 0 20px rgba(16,185,129,0.35);
    }

    /* =========================
    RIGHT WEATHER STATS
    ========================= */

    .weather-stats{
        margin-top:1rem;
    }

    .weather-stat-row{
        display:flex;

        justify-content:space-between;

        margin-bottom:1rem;

        color:white;
    }

    .weather-stat-label{
        color:#94a3b8;
    }

    /* =========================
    METRIC GRID
    ========================= */

    .metrics-grid{
        display:grid;

        grid-template-columns:repeat(4,1fr);

        gap:1rem;

        margin-top:1rem;
    }

    .metric-card{
        background:
            linear-gradient(
                135deg,
                rgba(15,35,75,0.95),
                rgba(5,15,40,0.95)
            );

        border:1px solid rgba(255,255,255,0.08);

        border-radius:22px;

        padding:1.8rem;

        text-align:center;

        transition:0.3s ease;
    }

    .metric-card:hover{
        transform: translateY(-5px);

        box-shadow:
            0 15px 35px rgba(0,0,0,0.25);
    }

    .metric-icon{
        font-size:3rem;
    }

    .metric-title{
        color:#cbd5e1;

        margin-top:1rem;

        font-size:1rem;
    }

    .metric-value{
        color:white;

        font-size:2.5rem;

        font-weight:800;

        margin-top:0.6rem;
    }

    .metric-sub{
        color:#94a3b8;
    }

    /* =========================
    SECTION CARDS
    ========================= */

    .section-card{
        background:
            linear-gradient(
                135deg,
                rgba(10,25,60,0.98),
                rgba(5,15,40,0.98)
            );

        border:1px solid rgba(255,255,255,0.08);

        border-radius:24px;

        padding:1.8rem;

        margin-top:1.5rem;
    }

    /* =========================
    TABLES
    ========================= */

    [data-testid="stDataFrame"]{
        border-radius:20px;
        overflow:hidden;
    }

    /* =========================
    ANIMATION
    ========================= */

    @keyframes fadeUp{
        from{
            opacity:0;
            transform: translateY(20px);
        }

        to{
            opacity:1;
            transform: translateY(0);
        }
    }

    /* =========================
    MOBILE
    ========================= */

    @media(max-width:900px){

        .metrics-grid{
            grid-template-columns:repeat(2,1fr);
        }

        .weather-temp{
            font-size:4rem;
        }

        .weather-center{
            flex-direction:column;
        }
    }

    </style>
    """, unsafe_allow_html=True)

    # =========================================================
    # HEADER
    # =========================================================

    st.markdown("""
    <div class=\"weather-wrapper\">

    <div class=\"weather-header\">
        <div style=\"display:flex; align-items:center; gap:1rem;\">
            <div style=\"font-size:3rem;\">🌤️</div>
            <div>
                <div class=\"weather-header-title\">
                    Weather Dashboard
                </div>
                <div class=\"weather-header-sub\">
                    Real-time weather data and 5-day forecasts
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # =========================================================
    # SEARCH AREA
    # =========================================================

    st.markdown('<div class="search-card">', unsafe_allow_html=True)

    search_col1, search_col2 = st.columns([3,1])
    with search_col1:
        city = st.text_input(
            "🌍 Enter City Name",
            placeholder="e.g. Pune, Mumbai, New York..."
        )
    with search_col2:
        units = st.selectbox(
            "Units",
            ["°C", "°F"]
        )
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        current_btn = st.button("☀️ Current Weather", use_container_width=True)
    with btn_col2:
        forecast_btn = st.button("📅 5-Day Forecast", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # =========================================================
    # FUNCTIONS
    # =========================================================
    def format_temp(temp):
        if units == "°F":
            return f"{(temp * 9/5) + 32:.0f}°F"
        return f"{temp:.0f}°C"

    def get_weather(city):
        r = requests.get(
            f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        )
        if r.status_code == 200:
            data = r.json()
            return {
                "temp": data["main"]["temp"],
                "feels_like": data["main"]["feels_like"],
                "humidity": data["main"]["humidity"],
                "pressure": data["main"]["pressure"],
                "wind_speed": data["wind"]["speed"],
                "condition": data["weather"][0]["description"].title(),
                "clouds": data["clouds"]["all"],
                "visibility": data.get("visibility",0)/1000,
                "rain_1h": data.get("rain",{}).get("1h",0),
                "rain_3h": data.get("rain",{}).get("3h",0),
                "country": data["sys"]["country"]
            }
        return None

    def get_forecast(city):
        r = requests.get(
            f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={API_KEY}&units=metric"
        )
        if r.status_code == 200:
            data = r.json()
            df = pd.DataFrame([{
                "datetime": i["dt_txt"],
                "temperature": i["main"]["temp"]
            } for i in data["list"]])
            return df
        return None

    # =========================================================
    # MAIN WEATHER CARD
    # =========================================================
    if current_btn and city.strip():
        weather = get_weather(city)
        if weather:
            st.markdown(f"""
            <div class=\"main-weather-card\">
                <div style='display: flex; justify-content: space-between;'>
                    <div class=\"weather-location\">📍 {city.title()}</div>
                    <div class=\"live-badge\">LIVE</div>
                </div>
                <div class=\"weather-country\">{weather['country']}</div>
                <div class=\"weather-center\">
                    <div class=\"weather-icon\">🌤️</div>
                    <div>
                        <div class=\"weather-temp\">{format_temp(weather['temp'])}</div>
                        <div class=\"weather-condition\">{weather['condition']}</div>
                    </div>
                </div>
                <div class=\"weather-stats\">
                    <div class=\"weather-stat-row\">
                        <span class=\"weather-stat-label\">Feels Like</span>
                        <span>{format_temp(weather['feels_like'])}</span>
                    </div>
                    <div class=\"weather-stat-row\">
                        <span class=\"weather-stat-label\">Humidity</span>
                        <span>{weather['humidity']}%</span>
                    </div>
                    <div class=\"weather-stat-row\">
                        <span class=\"weather-stat-label\">Wind Speed</span>
                        <span>{weather['wind_speed']} m/s</span>
                    </div>
                    <div class=\"weather-stat-row\">
                        <span class=\"weather-stat-label\">Pressure</span>
                        <span>{weather['pressure']} hPa</span>
                    </div>
                    <div class=\"weather-stat-row\">
                        <span class=\"weather-stat-label\">Visibility</span>
                        <span>{weather['visibility']:.1f} km</span>
                    </div>
                    <div class=\"weather-stat-row\">
                        <span class=\"weather-stat-label\">Cloud Cover</span>
                        <span>{weather['clouds']}%</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            # METRIC CARDS
            st.markdown(f"""
            <div class=\"metrics-grid\">
                <div class=\"metric-card\">
                    <div class=\"metric-icon\">🌡️</div>
                    <div class=\"metric-title\">Feels Like</div>
                    <div class=\"metric-value\">{format_temp(weather['feels_like'])}</div>
                </div>
                <div class=\"metric-card\">
                    <div class=\"metric-icon\">💧</div>
                    <div class=\"metric-title\">Humidity</div>
                    <div class=\"metric-value\">{weather['humidity']}%</div>
                </div>
                <div class=\"metric-card\">
                    <div class=\"metric-icon\">🌬️</div>
                    <div class=\"metric-title\">Wind Speed</div>
                    <div class=\"metric-value\">{weather['wind_speed']}</div>
                    <div class=\"metric-sub\">m/s</div>
                </div>
                <div class=\"metric-card\">
                    <div class=\"metric-icon\">⏲️</div>
                    <div class=\"metric-title\">Pressure</div>
                    <div class=\"metric-value\">{weather['pressure']}</div>
                    <div class=\"metric-sub\">hPa</div>
                </div>
                <div class=\"metric-card\">
                    <div class=\"metric-icon\">☁️</div>
                    <div class=\"metric-title\">Cloud Cover</div>
                    <div class=\"metric-value\">{weather['clouds']}%</div>
                </div>
                <div class=\"metric-card\">
                    <div class=\"metric-icon\">👁️</div>
                    <div class=\"metric-title\">Visibility</div>
                    <div class=\"metric-value\">{weather['visibility']:.1f}</div>
                    <div class=\"metric-sub\">km</div>
                </div>
                <div class=\"metric-card\">
                    <div class=\"metric-icon\">🌧️</div>
                    <div class=\"metric-title\">Rain (1h)</div>
                    <div class=\"metric-value\">{weather['rain_1h']}</div>
                    <div class=\"metric-sub\">mm</div>
                </div>
                <div class=\"metric-card\">
                    <div class=\"metric-icon\">🌦️</div>
                    <div class=\"metric-title\">Rain (3h)</div>
                    <div class=\"metric-value\">{weather['rain_3h']}</div>
                    <div class=\"metric-sub\">mm</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            # Save search to history
            now = datetime.now().strftime("%b %d, %Y %I:%M %p")
            new_row = pd.DataFrame({
                "Date & Time": [now],
                "City": [city.title()],
                "Temperature (°C)": [weather['temp']],
                "Condition": [weather['condition']]
            })
            if os.path.exists(HISTORY_FILE):
                history_df = pd.read_csv(HISTORY_FILE)
                history_df = pd.concat([history_df, new_row], ignore_index=True)
            else:
                history_df = new_row
            history_df.tail(100).to_csv(HISTORY_FILE, index=False)
        else:
            st.warning("City not found or API error.")
    elif current_btn:
        st.info("Enter a city name to get weather data.")

    # =========================================================
    # FORECAST
    # =========================================================
    if forecast_btn and city.strip():
        df = get_forecast(city)
        if df is not None:
            st.markdown("""
            <div class=\"section-card\">
                <div class=\"weather-header-title\">
                            📈 5-Day Forecast
                        </div>
            </div>
            """, unsafe_allow_html=True)
            chart = (
                alt.Chart(df)
                .mark_line(
                    point=True,
                    strokeWidth=4,
                    color="#10b981"
                )
                .encode(
                    x=alt.X(
                        "datetime:T",
                        axis=alt.Axis(
                            labelColor="white",
                            titleColor="white",
                            gridColor="#334155"
                        )
                    ),
                    y=alt.Y(
                        "temperature:Q",
                        axis=alt.Axis(
                            labelColor="white",
                            titleColor="white",
                            gridColor="#334155"
                        )
                    )
                )
                .properties(
                    height=400,
                    background="#071225"
                )
                .configure_view(
                    strokeWidth=0
                )
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.warning("City not found or API error.")
    elif forecast_btn:
        st.info("Enter a city name to get forecast data.")

    # =========================================================
    # SEARCH HISTORY

    
    # =========================================================
    st.markdown("""
    <div class=\"section-card\">
        <div class=\"weather-header-title\">
                    🕘 Search History 
                </div>
    </div>
    """, unsafe_allow_html=True)
    if os.path.exists(HISTORY_FILE):
        history_df = pd.read_csv(HISTORY_FILE)
        if not history_df.empty:
            st.dataframe(
                history_df.tail(10),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No search history yet.")
    else:
        st.info("No search history yet.")
    st.markdown("</div>", unsafe_allow_html=True)