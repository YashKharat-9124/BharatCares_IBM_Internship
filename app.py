
import streamlit as st
import pickle
import numpy as np

MODEL_PATH = "random_forest_churn_model.pkl"
SCALER_PATH = "scaler.pkl"


@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)
    with open(SCALER_PATH, "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler


model, scaler = load_artifacts()

st.set_page_config(
    page_title="Customer Churn Prediction System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------------
# UI CSS
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
        .stApp {
            background: #f7f9fc;
        }

        /* Hide default Streamlit chrome */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        .hero {
            background: linear-gradient(110deg, #2563eb 0%, #4f46e5 38%,
                        #8b2be2 68%, #ec1976 100%);
            padding: 24px 30px;
            border-radius: 0 0 18px 18px;
            color: white;
            margin: -1rem -1rem 24px -1rem;
            box-shadow: 0 8px 24px rgba(37, 99, 235, .16);
        }

        .hero-grid {
            display: grid;
            grid-template-columns: 1fr 1.35fr;
            gap: 25px;
            align-items: center;
        }

        .hero-title {
            font-size: 32px;
            line-height: 1.12;
            font-weight: 850;
            margin: 0;
        }

        .hero-title .yellow {
            color: #fde047;
        }

        .hero-subtitle {
            margin: 8px 0 0;
            color: #eef2ff;
            font-size: 14px;
        }

        .hero-points {
            display: flex;
            justify-content: flex-end;
            gap: 20px;
        }

        .hero-point {
            display: flex;
            gap: 10px;
            align-items: center;
            min-width: 135px;
        }

        .hero-icon {
            width: 45px;
            height: 45px;
            border: 1px solid rgba(255,255,255,.55);
            border-radius: 11px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 22px;
            background: rgba(255,255,255,.10);
        }

        .hero-point b {
            display: block;
            font-size: 14px;
        }

        .hero-point span {
            display: block;
            font-size: 11px;
            color: #f3e8ff;
            margin-top: 3px;
        }

        .panel {
            background: #ffffff;
            border: 1px solid #e6ebf5;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 5px 18px rgba(15, 23, 42, .055);
        }

        .panel-title {
            color: #17356f;
            font-size: 19px;
            font-weight: 800;
            margin: 0 0 5px;
        }

        .panel-line {
            height: 2px;
            width: 125px;
            background: #2563eb;
            margin-bottom: 18px;
        }

        .field-label {
            color: #1261d6;
            font-weight: 750;
            font-size: 14px;
            margin: 3px 0 7px;
        }

        /* Make Streamlit input labels blue instead of blending into white */
        div[data-testid="stNumberInput"] label,
        div[data-testid="stSelectbox"] label {
            color: #075fd6 !important;
            font-weight: 750 !important;
        }

        div[data-testid="stNumberInput"] input {
            color: #172033 !important;
            background: #ffffff !important;
        }

        div[data-baseweb="select"] > div {
            background: #ffffff !important;
            color: #172033 !important;
            border-color: #d8e1ef !important;
        }

        .predict-note {
            background: linear-gradient(90deg, #eff6ff, #faf5ff);
            border: 1px solid #ddd6fe;
            color: #27358a;
            border-radius: 11px;
            padding: 13px 15px;
            font-size: 13px;
            margin-top: 12px;
        }

        .live {
            float: right;
            color: #059669;
            background: #ecfdf5;
            border: 1px solid #bbf7d0;
            padding: 6px 11px;
            border-radius: 9px;
            font-size: 12px;
            font-weight: 700;
        }

        .prediction {
            background: linear-gradient(90deg, #ecfdf5, #f0fdf4);
            border: 1px solid #c8f0d7;
            border-radius: 12px;
            padding: 17px;
            text-align: center;
            margin-bottom: 15px;
        }

        .prediction.good {
            color: #059669;
        }

        .prediction.bad {
            background: linear-gradient(90deg, #fff1f2, #fff7ed);
            border-color: #fecdd3;
            color: #e11d48;
        }

        .prediction-main {
            font-size: 24px;
            font-weight: 850;
        }

        .prediction-sub {
            font-size: 13px;
            margin-top: 4px;
            font-weight: 650;
        }

        .metric-box {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 15px 10px;
            text-align: center;
        }

        .metric-label {
            color: #17356f;
            font-size: 13px;
            font-weight: 750;
        }

        .metric-value {
            font-size: 26px;
            font-weight: 850;
            margin-top: 5px;
        }

        .green { color: #0aa86b; }
        .blue { color: #0878e8; }
        .pink { color: #e11d75; }

        .confidence {
            margin-top: 15px;
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 15px;
        }

        .confidence-head {
            display: flex;
            justify-content: space-between;
            color: #17356f;
            font-weight: 750;
            font-size: 13px;
            margin-bottom: 8px;
        }

        .confidence-bar {
            height: 10px;
            border-radius: 20px;
            background: #e5e7eb;
            overflow: hidden;
        }

        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #16a86d, #36b37e);
            border-radius: 20px;
        }

        .snapshot-title {
            color: #17356f;
            font-size: 17px;
            font-weight: 800;
            margin: 18px 0 10px;
        }

        .snapshot {
            padding: 11px 13px;
            border-radius: 9px;
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            border: 1px solid rgba(0,0,0,.035);
        }

        .snapshot b {
            color: #17356f;
        }

        .snap-blue { background: #eff6ff; }
        .snap-orange { background: #fff7ed; }
        .snap-green { background: #ecfdf5; }
        .snap-pink { background: #fff1f2; }
        .snap-purple { background: #f5f3ff; }
        .snap-cyan { background: #ecfeff; }

        .footer {
            text-align: center;
            color: #64748b;
            padding: 20px;
            font-size: 12px;
        }

        div.stButton > button {
            min-height: 47px;
            border-radius: 10px;
            font-weight: 800;
            font-size: 14px;
        }

        @media (max-width: 900px) {
            .hero-grid { grid-template-columns: 1fr; }
            .hero-points { justify-content: flex-start; flex-wrap: wrap; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.markdown(
    """
    <div class="hero"><div class="hero-grid"><div><div class="hero-title">Customer Churn<br><span class="yellow">Prediction System</span></div><div class="hero-subtitle">AI-powered customer retention and churn risk assessment</div></div><div class="hero-points"><div class="hero-point"><div class="hero-icon">🤖</div><div><b>Machine Learning</b><span>Random Forest Algorithm</span></div></div><div class="hero-point"><div class="hero-icon">📈</div><div><b>Risk Analysis</b><span>Predict Churn Probability</span></div></div><div class="hero-point"><div class="hero-icon">👥</div><div><b>Customer Insights</b><span>Data Driven Decisions</span></div></div></div></div></div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Main columns
# ------------------------------------------------------------------
left, right = st.columns([1.45, 1], gap="large")

# ------------------------------------------------------------------
# Customer information
# ------------------------------------------------------------------
with left:
    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">👤 Customer Information</div>
            <div class="panel-line"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    a, b, c = st.columns(3)

    with a:
        credit_score = st.number_input(
            "💳 Credit Score", min_value=300, max_value=900, value=600, step=1
        )
        balance = st.number_input(
            "💰 Account Balance",
            min_value=0.0,
            value=50000.0,
            step=1000.0,
            format="%.2f",
        )
        active_member_label = st.selectbox(
            "👤 Active Member?", ["Yes", "No"]
        )
        gender = st.selectbox("⚥ Gender", ["Male", "Female"])

    with b:
        age = st.number_input("🎂 Age", min_value=18, max_value=100, value=30, step=1)
        num_products = st.number_input(
            "🛍️ Number of Products", min_value=1, max_value=4, value=2, step=1
        )
        salary = st.number_input(
            "💵 Estimated Salary",
            min_value=0.0,
            value=50000.0,
            step=1000.0,
            format="%.2f",
        )

    with c:
        tenure = st.number_input(
            "◷ Tenure (Years)", min_value=0, max_value=20, value=5, step=1
        )
        has_card_label = st.selectbox(
            "💳 Has Credit Card?", ["Yes", "No"]
        )
        country = st.selectbox("🌐 Country", ["France", "Germany", "Spain"])

    has_card = 1 if has_card_label == "Yes" else 0
    active_member = 1 if active_member_label == "Yes" else 0
    germany = 1 if country == "Germany" else 0
    spain = 1 if country == "Spain" else 0
    male = 1 if gender == "Male" else 0

    st.markdown("<br>", unsafe_allow_html=True)

    pcol, rcol = st.columns([1.1, 1])

    with pcol:
        predict_clicked = st.button(
            "🚀 Predict Churn", type="primary", use_container_width=True
        )

    with rcol:
        reset_clicked = st.button(
            "↻ Reset Inputs", use_container_width=True
        )

    if reset_clicked:
        st.rerun()

    st.markdown(
        """
        <div class="predict-note">
            ⓘ Provide customer details and click <b>"Predict Churn"</b> to see the result.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------
# Result panel
# ------------------------------------------------------------------
with right:
    st.markdown(
        """
        <div class="panel-title">
            🎯 Prediction Result
            <span class="live">● Live Prediction</span>
        </div>
        <div class="panel-line"></div>
        """,
        unsafe_allow_html=True,
    )

    prediction = None
    churn_probability = 0.0
    stay_probability = 0.0
    confidence = 0.0

    if predict_clicked:
        data = np.array(
            [[
                credit_score,
                age,
                tenure,
                balance,
                num_products,
                has_card,
                active_member,
                salary,
                germany,
                spain,
                male,
            ]],
            dtype=float,
        )

        try:
            scaled_data = scaler.transform(data)
            prediction = int(model.predict(scaled_data)[0])

            probabilities = model.predict_proba(scaled_data)[0]
            stay_probability = float(probabilities[0]) * 100
            churn_probability = float(probabilities[1]) * 100
            confidence = max(stay_probability, churn_probability)

        except Exception as error:
            st.error(f"Prediction failed: {error}")

    if prediction is None:
        st.markdown(
            """
            <div class="prediction good">
                <div class="prediction-main">🎯 Ready for Prediction</div>
                <div class="prediction-sub">Enter customer details and click Predict Churn</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif prediction == 1:
        st.markdown(
            """
            <div class="prediction bad">
                <div class="prediction-main">⚠️ Customer Will Exit</div>
                <div class="prediction-sub">High Churn Risk</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="prediction good">
                <div class="prediction-main">🛡️ Customer Will Stay</div>
                <div class="prediction-sub">Low Churn Risk</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    m1, m2 = st.columns(2)

    with m1:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Churn Probability</div>
                <div class="metric-value green">{churn_probability:.2f}%</div>
                <div style="font-size:12px;color:#64748b;">Risk of Leaving</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m2:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Stay Probability</div>
                <div class="metric-value blue">{stay_probability:.2f}%</div>
                <div style="font-size:12px;color:#0878e8;">Likely to Stay</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="confidence">
            <div class="confidence-head">
                <span>📊 Model Confidence</span>
                <span class="green">{confidence:.2f}%</span>
            </div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width:{confidence:.2f}%"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="snapshot-title">👤 Customer Snapshot</div>',
        unsafe_allow_html=True,
    )

    s1, s2 = st.columns(2)

    with s1:
        st.markdown(
            f'<div class="snapshot snap-blue"><b>💳 Credit Score</b><span>{credit_score}</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="snapshot snap-green"><b>💰 Account Balance</b><span>₹{balance:,.2f}</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="snapshot snap-purple"><b>👤 Active Member</b><span>{active_member_label}</span></div>',
            unsafe_allow_html=True,
        )

    with s2:
        st.markdown(
            f'<div class="snapshot snap-orange"><b>🎂 Age</b><span>{age} Years</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="snapshot snap-pink"><b>◷ Tenure</b><span>{tenure} Years</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="snapshot snap-cyan"><b>🛍️ Products</b><span>{num_products}</span></div>',
            unsafe_allow_html=True,
        )

st.markdown(
    """
    <div class="footer">
        Customer Churn Prediction System • Random Forest Machine Learning • Data-Driven Retention
    </div>
    """,
    unsafe_allow_html=True,
)
