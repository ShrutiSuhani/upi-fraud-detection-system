import streamlit as st
import pandas as pd
import joblib
import numpy as np
import re
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="UPI Fraud Detection System", page_icon="💳", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main-header { font-size: 2.8rem; color: #ffffff; background: transparent; text-align: center; margin-bottom: 0.5rem; font-weight: 900; }
    .pop-header { 
        font-size: 1.8rem; 
        color: #ffffff; 
        background: transparent;
        padding: 10px 0px; 
        margin-top: 1rem; 
        margin-bottom: 2rem; 
        font-weight: 800;
        border-bottom: 2px solid #444444;
    }
    .fraud-alert { background-color: #ffebee; padding: 1.5rem; border-radius: 0.5rem; border-left: 6px solid #d32f2f; margin: 1rem 0; color: #b71c1c; font-weight: bold; font-size: 1.2rem;}
    .legit-alert { background-color: #e8f5e9; padding: 1.5rem; border-radius: 0.5rem; border-left: 6px solid #2e7d32; margin: 1rem 0; color: #1b5e20; font-weight: bold; font-size: 1.2rem;}
    .metric-card { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; text-align: center; border: 1px solid #e0e0e0; }
    
    /* ENLARGE THE TAB HEADERS */
    button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
    }
    button[data-baseweb="tab"] {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD CACHED DATA & MODELS ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('upi_fraud_model.pkl')
        cols = joblib.load('model_columns.pkl')
        return model, cols
    except Exception as e:
        st.error(f"🚨 Missing .pkl files. Please ensure upi_fraud_model.pkl and model_columns.pkl are in the directory.")
        st.stop()

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('upi_transactions_2024.csv')
        if 'timestamp' in df.columns: df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except:
        return pd.DataFrame()

model, model_columns = load_assets()
db = load_data()

# For visualization mapping
vis_df = db.copy()
if 'fraud_flag' in vis_df.columns:
    vis_df.rename(columns={'fraud_flag': 'IsFraud'}, inplace=True)

# --- EXACT LABEL ENCODING MAPPINGS (from notebook) ---
TRANSACTION_TYPE_MAP  = {'Bill Payment': 0, 'P2M': 1, 'P2P': 2, 'Recharge': 3}
MERCHANT_CATEGORY_MAP = {'Education': 0, 'Entertainment': 1, 'Food': 2, 'Fuel': 3, 'Grocery': 4, 'Healthcare': 5, 'Other': 6, 'Shopping': 7, 'Transport': 8, 'Utilities': 9}
SENDER_STATE_MAP      = {'Andhra Pradesh': 0, 'Delhi': 1, 'Gujarat': 2, 'Karnataka': 3, 'Maharashtra': 4, 'Rajasthan': 5, 'Tamil Nadu': 6, 'Telangana': 7, 'Uttar Pradesh': 8, 'West Bengal': 9}
SENDER_BANK_MAP       = {'Axis': 0, 'HDFC': 1, 'ICICI': 2, 'IndusInd': 3, 'Kotak': 4, 'PNB': 5, 'SBI': 6, 'Yes Bank': 7}
RECEIVER_BANK_MAP     = {'Axis': 0, 'HDFC': 1, 'ICICI': 2, 'IndusInd': 3, 'Kotak': 4, 'PNB': 5, 'SBI': 6, 'Yes Bank': 7}
DAY_OF_WEEK_MAP       = {'Friday': 0, 'Monday': 1, 'Saturday': 2, 'Sunday': 3, 'Thursday': 4, 'Tuesday': 5, 'Wednesday': 6}

VALID_UPI_HANDLES = ['oksbi', 'okhdfc', 'okicici', 'okaxis', 'ybl', 'ibl', 'upi', 'paytm', 'apl', 'okhdfcbank', 'okicicibank']

# ================================================================
# LAYERS (From your original code)
# ================================================================
def validate_upi(upi_id):
    pattern = r'^[a-zA-Z0-9.\-_]+@[a-zA-Z]+$'
    if not re.match(pattern, upi_id): return False, "Invalid UPI ID format — must be name@bankhandle"
    handle = upi_id.split('@')[1].lower()
    if handle not in VALID_UPI_HANDLES: return False, f"Unrecognised UPI handle '@{handle}'"
    return True, ""

def history_check(sender_bank, sender_state, trans_type, amount):
    if db.empty: return False, "", None
    similar = db[(db['sender_bank'] == sender_bank) & (db['sender_state'] == sender_state) & (db['transaction type'] == trans_type)]
    if len(similar) < 10: return False, "", None
    avg_amount, std_amount, max_amount = similar['amount (INR)'].mean(), similar['amount (INR)'].std(), similar['amount (INR)'].max()
    threshold  = avg_amount + (3 * std_amount)
    stats = {'avg': round(avg_amount), 'max': round(max_amount), 'count': len(similar), 'threshold': round(threshold)}
    if amount > threshold: return True, f"Amount ₹{amount:,} is unusually high for similar transactions (avg ₹{round(avg_amount):,}, threshold ₹{round(threshold):,})", stats
    return False, "", stats

def velocity_check(sender_bank, sender_state, current_timestamp):
    if db.empty or 'timestamp' not in db.columns: return False, ""
    window_start = current_timestamp - pd.Timedelta(hours=1)
    recent = db[(db['sender_bank'] == sender_bank) & (db['sender_state'] == sender_state) & (db['timestamp'] >= window_start) & (db['timestamp'] <= current_timestamp)]
    count = len(recent)
    if count >= 3: return True, f"High velocity — {count} transactions from similar profile within 1 hour"
    return False, ""

def duplicate_check(transaction_id):
    if db.empty or 'transaction id' not in db.columns: return False, ""
    exists = transaction_id in db['transaction id'].values
    if exists: return True, f"Transaction ID '{transaction_id}' already exists — possible duplicate or replay attack"
    return False, ""

# ================================================================
# SIDEBAR & HEADER (Untouched)
# ================================================================
st.markdown('<div class="main-header">💳 UPI Fraud Detection System</div>', unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3774/3774298.png", width=100)
    st.markdown("## 🎯 About")
    st.info("**Real-time UPI Fraud Detection**\n\nThis system uses a Hybrid Rule Engine backed by an Undersampled Random Forest model to identify fraudulent patterns.")
    st.markdown("## ⚙️ Settings")
    ai_sensitivity = st.slider("AI Alert Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.1, help="Lower threshold = stricter security but more false alarms.")
    st.caption("Powered by Scikit-Learn & Streamlit")

# ================================================================
# TABS SETUP
# ================================================================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Explorer", "🤖 Model Performance", "🔮 Live Prediction", "📚 Architecture"])

# ================================================================
# TAB 1: DATA EXPLORER
# ================================================================
with tab1:
    st.markdown('<div class="pop-header">📈 Dataset Overview</div>', unsafe_allow_html=True)
    
    if not vis_df.empty and 'IsFraud' in vis_df.columns:
        # Key metrics perfectly aligned
        col1, col2, col3, col4 = st.columns(4)
        total_txns = len(vis_df)
        fraud_cases = vis_df['IsFraud'].sum()
        with col1: st.metric("Total Transactions", f"{total_txns:,}")
        with col2: st.metric("Fraud Cases", f"{fraud_cases:,}")
        with col3: st.metric("Fraud Rate", f"{(fraud_cases/total_txns)*100:.2f}%")
        with col4: st.metric("Features", len(model_columns))
        
        st.markdown("---")
        
        # Class Distribution explicitly titled
        st.markdown("### Class Distribution (The 520:1 Imbalance)")
        fig_class = make_subplots(rows=1, cols=2, subplot_titles=('Bar Chart (Log Scale)', 'Pie Chart'), specs=[[{'type': 'bar'}, {'type': 'pie'}]])
        class_counts = vis_df['IsFraud'].value_counts()
        labels = ['Legitimate', 'Fraudulent']
        colors = ['#4CAF50', '#FF5252']
        fig_class.add_trace(go.Bar(x=labels, y=class_counts.values, marker_color=colors, name='Count'), row=1, col=1)
        fig_class.add_trace(go.Pie(labels=labels, values=class_counts.values, marker_colors=colors, hole=0.3), row=1, col=2)
        fig_class.update_layout(showlegend=False, height=400, margin=dict(t=30, b=0, l=0, r=0))
        fig_class.update_yaxes(type="log", row=1, col=1) 
        st.plotly_chart(fig_class, use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Core Feature Distributions")
        c1, c2 = st.columns(2)
        with c1: 
            fig_amt = go.Figure()
            for f_class, color, label in [(0, '#4CAF50', 'Legit'), (1, '#FF5252', 'Fraud')]:
                subset = vis_df[vis_df['IsFraud'] == f_class]['amount (INR)']
                fig_amt.add_trace(go.Histogram(x=subset, nbinsx=100, name=label, marker_color=color, opacity=0.6, histnorm='probability density'))
            fig_amt.update_layout(title="Transaction Amount Distribution", xaxis_title="Amount (₹)", yaxis_title="Density", barmode='overlay', height=400)
            st.plotly_chart(fig_amt, use_container_width=True)
            
        with c2: 
            hour_col = 'hour_of_day' if 'hour_of_day' in vis_df.columns else 'hour' if 'hour' in vis_df.columns else None
            if hour_col: 
                fig_hr = go.Figure()
                for f_class, color, label in [(0, '#4CAF50', 'Legit'), (1, '#FF5252', 'Fraud')]:
                    subset = vis_df[vis_df['IsFraud'] == f_class][hour_col]
                    fig_hr.add_trace(go.Histogram(x=subset, nbinsx=24, name=label, marker_color=color, opacity=0.6, histnorm='probability density'))
                fig_hr.update_layout(title="Transaction Hour Distribution", xaxis_title="Hour", yaxis_title="Density", barmode='overlay', height=400)
                st.plotly_chart(fig_hr, use_container_width=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Advanced Transaction Patterns")
        c3, c4 = st.columns(2)
        with c3:
            if 'merchant_category' in vis_df.columns:
                fraud_by_merchant = vis_df.groupby('merchant_category')['IsFraud'].mean().sort_values(ascending=False)
                fig_merch = go.Figure(go.Bar(x=fraud_by_merchant.index, y=fraud_by_merchant.values, marker_color='#FF9800'))
                fig_merch.update_layout(title="Fraud Rate by Merchant Category", height=400)
                st.plotly_chart(fig_merch, use_container_width=True)
        with c4:
            if 'sender_state' in vis_df.columns:
                state_impact = vis_df[vis_df['IsFraud']==1]['sender_state'].value_counts().head(10)
                fig_state = go.Figure(go.Bar(x=state_impact.index, y=state_impact.values, marker_color='#9C27B0'))
                fig_state.update_layout(title="Fraud Count by State (Top 10)", height=400)
                st.plotly_chart(fig_state, use_container_width=True)
        
        with st.expander("View Raw Data Sample"):
            st.dataframe(vis_df.head(100), use_container_width=True)
    else:
        st.warning("Valid dataset required for Explorer view.")

# ================================================================
# TAB 2: MODEL PERFORMANCE
# ================================================================
with tab2:
    st.markdown('<div class="pop-header">🤖 Model Performance Metrics</div>', unsafe_allow_html=True)
    
    st.markdown("### Performance Comparison (Undersampled Dataset)")
    metrics_data = {
        'Model': ['Random Forest (Production)', 'Decision Tree', 'Gradient Boosting', 'XGBoost'],
        'Accuracy': [0.4427, 0.5104, 0.4740, 0.4896],
        'Precision': [0.4949, 0.5566, 0.5287, 0.5455],
        'Recall': [0.4623, 0.5566, 0.4340, 0.4528],
        'F1-Score': [0.4780, 0.5566, 0.4767, 0.4948]
    }
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df.set_index('Model').style.format("{:.2%}"), use_container_width=True)
    
    # Plotly Comparison Chart
    fig_comp = go.Figure()
    metrics_to_plot = ['Precision', 'Recall', 'F1-Score']
    for model_name in metrics_df['Model']:
        model_data = metrics_df[metrics_df['Model'] == model_name]
        fig_comp.add_trace(go.Bar(
            name=model_name, x=metrics_to_plot,
            y=[model_data[m].values[0] for m in metrics_to_plot],
            text=[f"{model_data[m].values[0]:.3f}" for m in metrics_to_plot], textposition='auto'
        ))
    fig_comp.update_layout(title="Algorithm Comparison", barmode='group', height=400, margin=dict(t=40, b=0, l=0, r=0))
    st.plotly_chart(fig_comp, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Confusion Matrix")
        fig_cm, ax = plt.subplots(figsize=(8, 6))
        cm = np.array([[39, 47], [47, 59]]) # Derived from notebook evaluation
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Legitimate', 'Fraudulent'], yticklabels=['Legitimate', 'Fraudulent'], ax=ax)
        ax.set_title('Random Forest - Undersampled Test Set')
        st.pyplot(fig_cm)
    
    with col2:
        st.markdown("### Feature Importance")
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            fig_feat, ax2 = plt.subplots(figsize=(10, 6))
            sorted_idx = np.argsort(importance)[-10:]
            bars = ax2.barh([model_columns[i] for i in sorted_idx], importance[sorted_idx], color='steelblue')
            ax2.set_xlabel('Coefficient Magnitude')
            ax2.set_title('Top 10 Drivers of Fraud Prediction')
            st.pyplot(fig_feat)
        else:
            st.info("Feature importance not available.")

    st.markdown("---")
    st.markdown("### 🎯 Feature Importance Interpretation")
    st.info("""
    The Random Forest feature importance chart reveals the structural decision-making process of the model. 
    * **Primary Drivers:** The transaction `amount (INR)` and `hour` consistently hold the highest coefficient magnitude. This aligns with our EDA, proving that abnormal spending volumes during unusual hours are the strongest pure behavioral indicators of fraud.
    * **Secondary Drivers:** Contextual features such as `merchant_category` and the `sender_bank` provide necessary routing context, allowing the model to differentiate between a late-night hospital bill (safe) vs. a late-night high-value P2P transfer (high risk).
    """)

# ================================================================
# TAB 3: LIVE PREDICTION
# ================================================================
with tab3:
    st.markdown("Enter transaction details to analyze risk.")
    st.divider()

    # --- WRAPPING INPUTS IN A FORM ---
    with st.form("transaction_scanner_form"):
        col1, col2 = st.columns(2)
        with col1:
            transaction_id = st.text_input("Transaction ID", value="TXN9999999999")
            sender_name    = st.text_input("Sender Name", value="John Doe")
            sender_upi     = st.text_input("Sender UPI ID", value="john@oksbi")
        with col2:
            receiver_name  = st.text_input("Receiver Name", value="Jane Doe")
            receiver_upi   = st.text_input("Receiver UPI ID", value="jane@okhdfc")

        st.divider()

        col3, col4 = st.columns(2)
        with col3:
            amount        = st.number_input("Transaction Amount (₹)", min_value=0, value=100000, step=1000)
            hour          = st.slider("Hour of Day (24h)", 0, 23, 3)
            day_of_week   = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        with col4:
            trans_type    = st.selectbox("Transaction Type", list(TRANSACTION_TYPE_MAP.keys()))
            merchant_cat  = st.selectbox("Merchant Category", list(MERCHANT_CATEGORY_MAP.keys()))
            sender_bank   = st.selectbox("Sender Bank", list(SENDER_BANK_MAP.keys()))
            receiver_bank = st.selectbox("Receiver Bank", list(RECEIVER_BANK_MAP.keys()))
            sender_state  = st.selectbox("Sender State", list(SENDER_STATE_MAP.keys()))

        # Change st.button to st.form_submit_button
        submitted = st.form_submit_button("🔍 Scan Transaction", type="primary")

    # --- PROCESSING HAPPENS ONLY AFTER SUBMIT IS CLICKED ---
    if submitted:
        now        = datetime.now()
        day        = now.day
        month      = now.month
        year       = now.year
        is_weekend = 1 if day_of_week in ["Saturday", "Sunday"] else 0
        dow_encoded = DAY_OF_WEEK_MAP.get(day_of_week, 1)

        device_type_encoded  = 0  
        network_type_encoded = 1  
        sender_age_encoded   = 1  
        receiver_age_encoded = 1  

        input_data = {
            'transaction type':   TRANSACTION_TYPE_MAP[trans_type],
            'merchant_category':  MERCHANT_CATEGORY_MAP[merchant_cat],
            'amount (INR)':       amount,
            'sender_age_group':   sender_age_encoded,
            'receiver_age_group': receiver_age_encoded,
            'sender_state':       SENDER_STATE_MAP[sender_state],
            'sender_bank':        SENDER_BANK_MAP[sender_bank],
            'receiver_bank':      RECEIVER_BANK_MAP[receiver_bank],
            'device_type':        device_type_encoded,
            'network_type':       network_type_encoded,
            'hour_of_day':        hour,
            'day_of_week':        dow_encoded,
            'is_weekend':         is_weekend,
            'hour':               hour,
            'day':                day,
            'month':              month,
            'year':               year
        }

        input_df       = pd.DataFrame([input_data], columns=model_columns)
        ai_probability = model.predict_proba(input_df)[0][1]

        sender_upi_valid,   sender_upi_reason   = validate_upi(sender_upi)
        receiver_upi_valid, receiver_upi_reason = validate_upi(receiver_upi)
        is_duplicate,       duplicate_reason    = duplicate_check(transaction_id)
        
        type_val = TRANSACTION_TYPE_MAP[trans_type] if 'transaction type' not in db.columns else trans_type
        history_flag, history_reason, history_stats = history_check(sender_bank, sender_state, type_val, amount)
        velocity_flag, velocity_reason = velocity_check(sender_bank, sender_state, pd.Timestamp(datetime(year, month, day, hour)))

        rule_flag         = False
        rule_reason       = ""
        manual_risk_score = 0.0

        if hour < 5 and amount > 50000:
            rule_flag         = True
            manual_risk_score = 1.0
            rule_reason       = "CRITICAL: High Value Transaction during Night (12 AM – 5 AM)"
        elif hour < 5:
            manual_risk_score = 0.65
            rule_reason       = "Warning: Transaction during unusual hours."
        elif is_weekend and amount > 80000 and trans_type == "P2P":
            rule_flag         = True
            manual_risk_score = 1.0
            rule_reason       = "Unusual High Value P2P Transfer on Weekend"
        elif amount > 90000:
            manual_risk_score = 0.60
            rule_reason       = "Warning: Unusually high transaction amount."

        if is_duplicate:
            final_risk       = 1.0
            detection_source = "🛡️ Duplicate Transaction Check"
            rule_reason      = duplicate_reason
        elif not sender_upi_valid or not receiver_upi_valid:
            final_risk       = 1.0
            detection_source = "🛡️ UPI Validation"
            rule_reason      = sender_upi_reason if not sender_upi_valid else receiver_upi_reason
        elif rule_flag:
            final_risk       = 1.0
            detection_source = "🛡️ Expert Rule Engine"
        else:
            risk_scores = [ai_probability, manual_risk_score]
            if history_flag: risk_scores.append(0.75)
            if velocity_flag: risk_scores.append(0.80)
            final_risk = max(risk_scores)

            if velocity_flag:
                detection_source = "🛡️ Transaction Frequency Check"
                rule_reason      = velocity_reason
            elif history_flag:
                detection_source = "🛡️ Transaction History Check"
                rule_reason      = history_reason
            elif manual_risk_score > 0:
                detection_source = "🛡️ Hybrid (Rule + AI)"
            else:
                detection_source = "🤖 AI Model"

        st.divider()

        with st.expander("🧾 Transaction Summary", expanded=True):
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Transaction ID**\n\n{transaction_id}")
            c2.markdown(f"**Sender**\n\n{sender_name}\n\n{sender_upi}")
            c3.markdown(f"**Receiver**\n\n{receiver_name}\n\n{receiver_upi}")
            c4, c5, c6 = st.columns(3)
            c4.markdown(f"**Amount**\n\n₹{amount:,}")
            c5.markdown(f"**Type**\n\n{trans_type}")
            c6.markdown(f"**Time**\n\n{day_of_week}, {hour:02d}:00")

        with st.expander("🔎 Intelligence Summary", expanded=False):
            st.markdown(f"**🤖 AI Confidence:** The AI model estimates a `{ai_probability:.1%}` chance this transaction is fraudulent.")
            if sender_upi_valid and receiver_upi_valid:
                st.markdown("**📱 UPI IDs:** Both sender and receiver UPI IDs look genuine and follow the correct format.")
            else:
                problem = sender_upi_reason if not sender_upi_valid else receiver_upi_reason
                st.markdown(f"**📱 UPI IDs:** ⚠️ Something looks off — {problem}")

            if is_duplicate:
                st.markdown(f"**🔁 Transaction ID:** This ID (`{transaction_id}`) has been seen before in our records. This could mean the transaction is being replayed or tampered with.")
            else:
                st.markdown(f"**🔁 Transaction ID:** This is a fresh, unique transaction ID — not seen before.")

            if history_stats:
                if history_flag:
                    st.markdown(f"**📊 Amount Check:** This transaction (₹{amount:,}) is much higher than what {sender_bank} users in {sender_state} typically send via {trans_type} (usually around ₹{history_stats['avg']:,}). This is unusual.")
                else:
                    st.markdown(f"**📊 Amount Check:** The amount (₹{amount:,}) is within the normal range for {sender_bank} users in {sender_state} doing {trans_type} transfers (typical average is ₹{history_stats['avg']:,}).")
            else:
                st.markdown(f"**📊 Amount Check:** Not enough historical data for this profile to make a comparison.")

            if velocity_flag:
                st.markdown(f"**⚡ Transaction Frequency:** Multiple transactions were detected from a similar profile in a short time window. This rapid activity is a known pattern in fraud cases.")
            else:
                st.markdown("**⚡ Transaction Frequency:** No unusual activity detected — transaction frequency looks normal.")

        # DISPLAY BANNERS
        st.markdown("<br>", unsafe_allow_html=True)
        c_res1, c_res2 = st.columns([2, 1])
        with c_res1:
            if final_risk > 0.8:
                st.markdown('<div class="fraud-alert">⚠️ FRAUDULENT TRANSACTION DETECTED!</div>', unsafe_allow_html=True)
                if rule_reason: st.write(f"**Reason:** {rule_reason}")
                st.write(f"**Source:** {detection_source}")
            elif final_risk > 0.5:
                st.markdown('<div class="legit-alert" style="background-color:#fff3cd; border-left-color:#ffc107; color:#856404;">⚠️ SUSPICIOUS - REQUIRE OTP</div>', unsafe_allow_html=True)
                if rule_reason: st.write(f"**Note:** {rule_reason}")
                st.write(f"**Source:** {detection_source}")
            else:
                st.markdown('<div class="legit-alert">✅ LEGITIMATE TRANSACTION</div>', unsafe_allow_html=True)
                st.write(f"**Source:** {detection_source}")
        
        with c_res2:
            st.metric(label="Final Risk Score", value=f"{final_risk:.1%}")

# ================================================================
# TAB 4: ARCHITECTURE (Extracted from Notebook)
# ================================================================
with tab4:
    st.markdown("## Architectural Journey: The ML Accuracy Paradox")
    
    st.markdown("""
    Developing a robust UPI Fraud Detection system requires navigating extreme statistical anomalies. 
    This tab documents the rigorous experimental methodology applied to the initial 250,000 transaction dataset, explicitly detailing why standard Machine Learning approaches failed, and why a Hybrid Rule Engine + Undersampled ML architecture was ultimately required.
    """)
    
    st.markdown("---")
    st.markdown("### 1. The Imbalance & The Failed Experiments")
    st.markdown("""
    During the EDA phase, the dataset revealed an extreme **Class Imbalance Ratio of 520:1**. Legitimate traffic overwhelmingly drowned out the fraudulent signals. Statistical tests (Chi-Square/ANOVA) confirmed that fraud in this dataset *does not* follow obvious, easily separable patterns across standard features.
    
    To combat this, multiple standard data science techniques were tested and subsequently rejected:
    """)
    
    # Table extracted from the provided Jupyter Notebook snippet
    exp_data = {
        "Approach": ["SMOTE + Random Forest", "SMOTE + XGBoost", "Frequency Encoding + SMOTE", "Threshold Lowering (10%)", "Class Weights (Balanced)"],
        "Result": ["0% fraud recall", "0% fraud recall", "0% fraud recall", "0% fraud recall", "0% fraud recall"],
        "Why It Failed": [
            "Signal too weak on full 250k dataset",
            "Same root cause",
            "Imbalance ratio (520:1) too extreme",
            "Model had no fraud signal to threshold",
            "Features don't separate fraud statistically"
        ]
    }
    st.table(pd.DataFrame(exp_data))
    
    st.markdown("---")
    st.markdown("### 2. The Breakthrough: Undersampling")
    st.markdown("""
    Due to the failures above, a drastic **Undersampling** strategy was implemented. The majority class was reduced to perfectly match the minority class, resulting in a balanced training set of **480 fraud vs. 480 legitimate** transactions. 
    
    This forced the model to exclusively study the behavioral boundaries between legitimate and fraudulent transactions, rather than taking the statistically safe route of predicting 'Legitimate' 100% of the time.
    * **Result:** F1 Scores climbed from 0.00 to the 46-56% range.
    """)
    
    st.markdown("---")
    st.markdown("### 3. Final Model & Honest Assessment")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.info("""
        **Final ML Specifications**
        * **Algorithm:** Random Forest
        * **Hyperparameters:** 100 estimators
        * **Training Strategy:** Balanced Undersampling (480 vs 480)
        * **Features Used:** 14 columns
        * **Best F1 Score:** ~0.56 (Decision Tree baseline on 192-sample test set)
        * **Best PR-AUC:** 0.574 (Gradient Boosting)
        """)
        
    with col_b:
        st.warning("""
        **The Honest Assessment**
        All four tested models (DT, RF, GB, XGB) scored a Precision-Recall AUC between 0.54 and 0.57. This is only modestly above the 0.50 random baseline. 
        
        This is a direct, unavoidable consequence of the weak statistical feature-fraud associations identified early in the analysis. Machine Learning *alone* cannot definitively catch UPI fraud without high false positive rates.
        """)
        
    st.markdown("### 4. The Hybrid Solution")
    st.markdown("""
    Because the ML model alone lacks the required deterministic confidence, the system was expanded into a **Hybrid Architecture**.
    
    The **Rule Engine** component of the hybrid system compensates for the ML limitations by catching pattern-based, deterministic fraud (e.g., Velocity spikes, Impossible hours, Replay attacks, and Cohort deviations) that ML cannot detect from static features alone. The Random Forest acts as the final behavioral backstop.
    """)