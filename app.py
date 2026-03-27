import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
 
#  Optional heavy deps 
try:
    from tensorflow.keras.models import load_model as keras_load
except Exception:
    keras_load = None
 
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False
 
#  Page config 
st.set_page_config(
    page_title=" AI Popularity Predictor",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
#  Load External CSS 
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name, "r", encoding="utf-8") as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        st.warning(f"⚠️ ไม่พบไฟล์ CSS: {file_name}")

# เรียกใช้ง style.css
local_css("style.css")
 
# Helper utilities 
def load_safe(path):
    return joblib.load(path) if os.path.exists(path) else None
 
def load_nn(path):
    if keras_load is None:
        return None
    try:
        return keras_load(path)
    except Exception:
        return None

def get_actionable_insight(prob):
    """Returns verdict, insight text, and color theme based on probability."""
    if prob >= 0.75:
        return "🔥 ความนิยมสูงมาก", "🎯 **High Potential:** คอนเทนต์นี้มีองค์ประกอบที่ตลาดต้องการสูง แนะนำให้ลงทุนด้านการตลาดและการโปรโมทเพิ่มได้เลย", "success"
    elif prob >= 0.50:
        return "⚖️ ปานกลางถึงดี", "💡 **Balanced:** มีโอกาสเข้าถึงผู้คนได้ดี แต่อาจจะต้องลองปรับ 'ราคา' หรือ 'ช่วงเวลาปล่อย' เพื่อเพิ่มโอกาสทะยานสู่กลุ่มท็อป", "warning"
    else:
        return "📉 ความนิยมเฉพาะกลุ่ม", "🔍 **Niche Audience:** ข้อมูลระบุว่ากลุ่มเป้าหมายอาจจะเฉพาะทางเกินไป หรือมีการแข่งขันสูง แนะนำให้ทบทวนฟีเจอร์หลักอีกครั้ง", "error"
 
#  Load data (cached) 
@st.cache_data
def load_data():
    try:
        steam = pd.read_csv("data/steam_games_2026.csv")
    except Exception:
        np.random.seed(42)
        steam = pd.DataFrame({
            "Review_Score_Pct": np.random.normal(72, 15, 500).clip(0, 100),
            "Price_USD": np.random.exponential(15, 500).clip(0, 80),
            "Primary_Genre": np.random.choice(
                ["Action","Adventure","RPG","Strategy","Simulation","Sports","Puzzle"], 500),
        })
    try:
        netflix = pd.read_csv("data/netflix_titles.csv")
    except Exception:
        np.random.seed(7)
        netflix = pd.DataFrame({
            "release_year": np.random.randint(1990, 2024, 800),
            "type": np.random.choice(["Movie","TV Show"], 800),
            "duration": np.random.randint(20, 200, 800),
        })
    return steam, netflix
 
steam, netflix = load_data()
 
#  Load models 
steam_model  = load_safe("models/steam/ensemble.pkl")
steam_scaler = load_safe("models/steam/scaler.pkl")
steam_nn     = load_nn("models/steam/nn.keras")
steam_le     = load_safe("models/steam/label_encoder.pkl")
 
netflix_model  = load_safe("models/netflix/model.pkl")
netflix_scaler = load_safe("models/netflix/scaler.pkl")
netflix_nn     = load_nn("models/netflix/nn.keras")
netflix_le     = load_safe("models/netflix/type_encoder.pkl")
netflix_genre_columns = load_safe("models/netflix/genre_columns.pkl") # [เพิ่ม] โหลดรายชื่อ Genre
 
#  Sidebar 
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🔭 PopScope AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">Popularity Intelligence</div>', unsafe_allow_html=True)
 
    page = st.radio(
        "Navigation",
        options=[
            "📊  Overview",
            "🗄️  Datasets",
            "🔍  Features",
            "🧠  ML Models",
            "🤖  Neural Network",
            "🎮  Steam Predict",
            "🎬  Netflix Predict",
        ],
        label_visibility="collapsed",
    )
 
# PAGE: OVERVIEW
 
if page == "📊  Overview":
    st.markdown('<div class="page-title">Platform Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">ภาพรวมข้อมูลและสถิติจาก Steam & Netflix</div>', unsafe_allow_html=True)
 
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'''<div class="metric-tile">
            <div class="value" style="color:var(--accent-blue)">{len(steam):,}</div>
            <div class="label">🎮 Steam Games</div></div>''', unsafe_allow_html=True)
    with c2:
        avg_score = round(steam["Review_Score_Pct"].mean(), 1)
        st.markdown(f'''<div class="metric-tile">
            <div class="value" style="color:var(--accent-green)">{avg_score}</div>
            <div class="label">⭐ Avg Review %</div></div>''', unsafe_allow_html=True)
    with c3:
        st.markdown(f'''<div class="metric-tile">
            <div class="value" style="color:var(--accent-pink)">{len(netflix):,}</div>
            <div class="label">🎬 Netflix Titles</div></div>''', unsafe_allow_html=True)
    with c4:
        n_genres = steam["Primary_Genre"].nunique()
        st.markdown(f'''<div class="metric-tile">
            <div class="value" style="color:#c084fc">{n_genres}</div>
            <div class="label">🎯 Steam Genres</div></div>''', unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("### 🎮 Steam — Review Score Distribution")
        st.markdown('<p class="tooltip-label" style="margin-bottom:10px;">การกระจายตัวของคะแนนรีวิวจาก Steam (0–100%)</p>', unsafe_allow_html=True)
        if HAS_PLOTLY:
            fig = px.histogram(
                steam, x="Review_Score_Pct", nbins=30,
                color_discrete_sequence=["#4f8ef7"],
                template="plotly_dark",
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0,r=0,t=10,b=0), height=260,
                xaxis_title="Review Score (%)", yaxis_title="จำนวนเกม",
            )
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)
 
    with col_right:
        st.markdown("### 🎬 Netflix — Title by Release Year")
        st.markdown('<p class="tooltip-label" style="margin-bottom:10px;">จำนวน Title ที่ออกฉายในแต่ละปี</p>', unsafe_allow_html=True)
        if HAS_PLOTLY:
            fig2 = px.histogram(
                netflix, x="release_year", nbins=30,
                color_discrete_sequence=["#f75f8e"],
                template="plotly_dark",
            )
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0,r=0,t=10,b=0), height=260,
                xaxis_title="Release Year", yaxis_title="จำนวน Title",
            )
            fig2.update_traces(marker_line_width=0)
            st.plotly_chart(fig2, use_container_width=True)
 
    st.markdown("---")

    if HAS_PLOTLY and "Primary_Genre" in steam.columns:
        st.markdown("### 🎮 Steam — Top Genres by Title Count")
        st.markdown('<p class="tooltip-label" style="margin-bottom:10px;">แนวเกมที่มีจำนวนเกมมากที่สุดใน Dataset</p>', unsafe_allow_html=True)
        genre_counts = steam["Primary_Genre"].value_counts().head(10).reset_index()
        genre_counts.columns = ["Genre","Count"]
        fig3 = px.bar(genre_counts, x="Count", y="Genre", orientation="h",
                      color="Count", color_continuous_scale=["#1f2a40","#38d9a9"],
                      template="plotly_dark")
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0,r=0,t=10,b=0), height=280,
            showlegend=False, coloraxis_showscale=False,
        )
        st.plotly_chart(fig3, use_container_width=True)
 
 
# PAGE: DATASETS
 
elif page == "🗄️  Datasets":
    st.markdown('<div class="page-title">Dataset Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">ดูตัวอย่างข้อมูลดิบที่ใช้ฝึกโมเดล AI</div>', unsafe_allow_html=True)
 
    st.markdown('''<div class="card card-neutral">
    <strong>📥 แหล่งที่มาของข้อมูล</strong><br><br>
    <table style="width:100%; text-align:left; border-collapse:collapse;">
        <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);"><th style="padding:8px;">Dataset</th><th style="padding:8px;">แหล่งที่มา</th><th style="padding:8px;">จำนวนแถว</th><th style="padding:8px;">วัตถุประสงค์</th></tr>
        <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);"><td style="padding:8px;">🎮 Steam Games 2026</td><td style="padding:8px;">Kaggle</td><td style="padding:8px;">~500+</td><td style="padding:8px;">ทำนาย Popularity Score</td></tr>
        <tr><td style="padding:8px;">🎬 Netflix Titles</td><td style="padding:8px;">Kaggle</td><td style="padding:8px;">~800+</td><td style="padding:8px;">ทำนายความนิยมของ Content</td></tr>
    </table><br>
    <strong>🎯 เป้าหมาย</strong> — ใช้ข้อมูลเชิงตัวเลขและหมวดหมู่เพื่อ predict ว่า game / title นั้น <strong>จะได้รับความนิยมสูงหรือไม่</strong>
    </div>''', unsafe_allow_html=True)
 
    tab1, tab2 = st.tabs(["🎮 Steam Dataset", "🎬 Netflix Dataset"])
 
    with tab1:
        st.markdown(f'<span class="badge badge-blue">ROWS: {len(steam):,}</span>'
                    f'<span class="badge badge-blue">COLS: {len(steam.columns)}</span>',
                    unsafe_allow_html=True)
        st.dataframe(steam.head(10), use_container_width=True, height=320)
        st.markdown('<p class="tooltip-label">แสดง 10 แถวแรก — เลื่อนซ้าย/ขวาเพื่อดูคอลัมน์ทั้งหมด</p>', unsafe_allow_html=True)
 
    with tab2:
        st.markdown(f'<span class="badge badge-pink">ROWS: {len(netflix):,}</span>'
                    f'<span class="badge badge-pink">COLS: {len(netflix.columns)}</span>',
                    unsafe_allow_html=True)
        st.dataframe(netflix.head(10), use_container_width=True, height=320)
        st.markdown('<p class="tooltip-label">แสดง 10 แถวแรก — เลื่อนซ้าย/ขวาเพื่อดูคอลัมน์ทั้งหมด</p>', unsafe_allow_html=True)
 

# PAGE: FEATURES

elif page == "🔍  Features":
    st.markdown('<div class="page-title">Feature Engineering</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Features (ตัวแปรนำเข้า) ที่โมเดลใช้ในการเรียนรู้</div>', unsafe_allow_html=True)
 
    col1, col2 = st.columns(2)
 
    with col1:
        st.markdown('''<div class="card card-steam">
        <h3>🎮 Steam Features</h3>
        <div class="info-box"><strong>Price_USD</strong> — ราคาเกม (หน่วย: ดอลลาร์สหรัฐ)<br>งานวิจัยพบว่าช่วงราคา $0–$20 มักได้รับความนิยมสูงกว่า</div>
        <div class="info-box"><strong>Review_Score_Pct</strong> — % คะแนนรีวิวจาก Steam Community (0–100)<br>ยิ่งสูงยิ่งดี: ≥80% = Overwhelmingly Positive</div>
        <div class="info-box"><strong>Primary_Genre</strong> — แนวเกมหลัก เช่น Action, RPG, Strategy<br>แนวเกมมีผลต่อฐานผู้เล่น — Encode เป็นตัวเลขก่อนป้อนโมเดล</div>
        </div>''', unsafe_allow_html=True)
 
    with col2:
        st.markdown('''<div class="card card-netflix">
        <h3>🎬 Netflix Features</h3>
        <div class="info-box" style="border-left-color:var(--accent-pink)"><strong>type</strong> — ประเภทของ Content: Movie หรือ TV Show<br>TV Show มักสร้าง Engagement ยาวนานกว่า Movie</div>
        <div class="info-box" style="border-left-color:var(--accent-pink)"><strong>release_year</strong> — ปีที่ออกฉาย<br>Content ใหม่ (หลัง 2015) มีแนวโน้ม Engagement สูงกว่า</div>
        <div class="info-box" style="border-left-color:var(--accent-pink)"><strong>duration</strong> — ความยาว (นาที สำหรับ Movie / Episodes สำหรับ TV Show)<br>ความยาวที่เหมาะสมช่วยเพิ่ม Completion Rate</div>
        </div>''', unsafe_allow_html=True)
 
    st.markdown('''<div class="card card-neutral">
    <h3>🔄 Preprocessing Pipeline</h3>
    <ul>
        <li><strong>Label Encoding</strong> — แปลง text เป็นตัวเลข (เช่น "Action" → 0, "RPG" → 1)</li>
        <li><strong>Standard Scaler</strong> — ทำให้ทุก feature อยู่ในสเกลเดียวกัน (mean=0, std=1)</li>
        <li><strong>Output</strong> — ค่า Probability 0–100% แสดงโอกาสที่ title จะ "ได้รับความนิยมสูง"</li>
    </ul>
    </div>''', unsafe_allow_html=True)
 

# PAGE: ML MODELS

elif page == "🧠  ML Models":
    st.markdown('<div class="page-title">Machine Learning Models</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">โมเดล ML แบบ Ensemble ที่ใช้ทำนายความนิยม</div>', unsafe_allow_html=True)
 
    cols = st.columns(3)
    models_info = [
        ("🌲 Random Forest", "accent-blue",
         "สร้าง Decision Tree หลายต้นพร้อมกัน แล้ว Vote ผลลัพธ์ด้วยกัน ช่วยลด Overfitting",
         "ทนต่อ Noisy Data, ไม่ต้องปรับ Scale มาก, เห็น Feature Importance"),
        ("⚡ Gradient Boosting", "accent-green",
         "สร้าง Tree ทีละต้น โดยแต่ละต้นเรียนรู้จาก Error ของต้นก่อนหน้า (XGBoost-style)",
         "Accuracy สูง, จัดการ Imbalanced Data ได้ดี"),
        ("📈 Logistic Regression", "accent-pink",
         "โมเดล Linear ที่ใช้ Sigmoid Function แปลงผลเป็น Probability (0–1)",
         "รวดเร็ว, อธิบายได้ง่าย, เหมาะเป็น Baseline"),
    ]
    for col, (title, color, desc, pros) in zip(cols, models_info):
        with col:
            st.markdown(f'''<div class="card">
<div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;color:var(--{color});margin-bottom:10px">{title}</div>
<p style="font-size:0.88rem;color:var(--text-main);line-height:1.6">{desc}</p>
<hr style="border-color:var(--border-color);margin:12px 0">
<p style="font-size:0.8rem;color:var(--text-muted)">✅ {pros}</p>
</div>''', unsafe_allow_html=True)
 
    st.markdown('''<div class="card card-neutral">
    <h3>🔗 Ensemble Strategy</h3>
    <p>โปรเจกต์นี้ใช้ <strong>Soft Voting Ensemble</strong> — รวม Probability จากทุกโมเดลแล้วเฉลี่ย</p>
    <strong>ทำไมถึงดีกว่าใช้โมเดลเดียว?</strong>
    <ul>
        <li>แต่ละโมเดลมีจุดแข็งต่างกัน → รวมกันเพิ่ม Robustness</li>
        <li>ลด Variance (ป้องกัน Overfitting)</li>
        <li>โดยทั่วไปให้ Accuracy สูงกว่า Single Model ~2–5%</li>
    </ul>
    </div>''', unsafe_allow_html=True)
 

# PAGE: NEURAL Network

elif page == "🤖  Neural Network":
    st.markdown('<div class="page-title">Neural Network Architecture</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">โครงสร้าง Deep Learning ที่ใช้ใน PopScope AI</div>', unsafe_allow_html=True)
 
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('''<div class="card">
        <h3>⚙️ Training Settings</h3>
        <ul>
            <li><strong>Optimizer:</strong> Adam (lr=0.001)</li>
            <li><strong>Loss:</strong> Binary Crossentropy</li>
            <li><strong>Epochs:</strong> 50–100</li>
            <li><strong>Batch Size:</strong> 32</li>
            <li><strong>Validation:</strong> 20% split</li>
        </ul>
        </div>''', unsafe_allow_html=True)
 
    with col2:
        st.markdown('''<div class="card">
        <h3>📖 Key Concepts</h3>
        <p><strong>ReLU</strong> — ฟังก์ชัน Activation ที่ทำให้โมเดลเรียนรู้ Non-linear Patterns ได้</p>
        <p><strong>Dropout</strong> — สุ่มปิด Neuron บางส่วนระหว่าง Training เพื่อป้องกัน Overfitting</p>
        <p><strong>Sigmoid</strong> — แปลง output เป็น Probability 0–1 เหมาะสำหรับ Binary Classification</p>
        <p><strong>Adam</strong> — Optimizer ที่ปรับ Learning Rate อัตโนมัติ — รวดเร็วและ Stable</p>
        </div>''', unsafe_allow_html=True)
 
# PAGE: STEAM PREDICT# 

elif page == "🎮  Steam Predict":
    st.markdown('<div class="page-title">Steam Popularity Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">กรอกข้อมูลเกมเพื่อทำนายโอกาสที่จะได้รับความนิยม พร้อมรับคำแนะนำ</div>', unsafe_allow_html=True)
 
    st.markdown("### 📝 Input Features")
    st.markdown('<p class="tooltip-label" style="margin-bottom:20px;">ปรับแต่งพารามิเตอร์ด้านล่างเพื่อดูว่าส่งผลต่อโอกาสความสำเร็จอย่างไร</p>', unsafe_allow_html=True)
 
    col1, col2, col3 = st.columns(3)
    with col1:
        price = st.number_input(
            "💰 ราคาเกม (USD)",
            min_value=0.0, max_value=200.0, value=19.99, step=0.99,
            help="ราคาขายบน Steam เป็นดอลลาร์สหรัฐ เกม Free-to-Play ใส่ 0"
        )
        st.markdown('<p class="tooltip-label">💡 เกมราคา $0–$20 เข้าถึงผู้เล่นได้กว้างกว่า</p>', unsafe_allow_html=True)
 
    with col2:
        score = st.number_input(
            "⭐ Review Score (%)",
            min_value=0.0, max_value=100.0, value=75.0, step=1.0,
            help="% ของรีวิวที่เป็นบวก จาก Steam Community"
        )
        st.markdown('<p class="tooltip-label">💡 เป้าหมายคือรักษาฐานคะแนน >80% ขึ้นไป</p>', unsafe_allow_html=True)
 
    with col3:
        genres = list(steam["Primary_Genre"].dropna().unique()) if steam_le else [
            "Action","Adventure","RPG","Strategy","Simulation","Sports","Puzzle"
        ]
        genre = st.selectbox(
            "🎮 Primary Genre",
            options=genres,
            help="แนวเกมหลัก ใช้ในการจัดกลุ่มและวิเคราะห์"
        )
        st.markdown('<p class="tooltip-label">💡 แนวเกมมีผลต่อขนาดของฐานผู้เล่นตั้งต้น</p>', unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    col_ml, col_nn = st.columns(2)
    models_ready = steam_model and steam_scaler and steam_le
 
    with col_ml:
        if st.button("🧠 Predict with ML Ensemble", use_container_width=True):
            if not models_ready:
                st.warning("⚠️ ไม่พบไฟล์โมเดล ML — กรุณาฝึกโมเดลก่อน (models/steam/ensemble.pkl)")
            else:
                try:
                    g_enc = steam_le.transform([genre])[0]
                    data = steam_scaler.transform(np.array([[price, score, g_enc]]))
                    prob = steam_model.predict_proba(data)[0][1]
                    verdict, insight, theme = get_actionable_insight(prob)
                    
                    st.markdown(f'''<div class="result-box">
<div class="result-percent">{prob*100:.1f}%</div>
<div class="result-label">ML Ensemble Confidence Score</div>
<br><span style="font-size:1.3rem; font-weight:700;">{verdict}</span>
</div>''', unsafe_allow_html=True)
                    st.progress(float(prob))
                    
                    if theme == "success": st.success(insight)
                    elif theme == "warning": st.warning(insight)
                    else: st.error(insight)
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาด: {e}")
 
    with col_nn:
        if st.button("🤖 Predict with Neural Net", use_container_width=True):
            if not (models_ready and steam_nn):
                st.warning("⚠️ ไม่พบไฟล์ Neural Network — กรุณาฝึกโมเดลก่อน (models/steam/nn.keras)")
            else:
                try:
                    g_enc = steam_le.transform([genre])[0]
                    data = steam_scaler.transform(np.array([[price, score, g_enc]]))
                    prob = float(steam_nn.predict(data, verbose=0)[0][0])
                    verdict, insight, theme = get_actionable_insight(prob)
                    
                    st.markdown(f'''<div class="result-box" style="border-color: rgba(56,217,169,0.3);">
<div class="result-percent" style="background: linear-gradient(to right, var(--accent-green), var(--accent-blue)); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{prob*100:.1f}%</div>
<div class="result-label">Neural Network Confidence Score</div>
<br><span style="font-size:1.3rem; font-weight:700;">{verdict}</span>
</div>''', unsafe_allow_html=True)
                    st.progress(float(prob))
                    
                    if theme == "success": st.success(insight)
                    elif theme == "warning": st.warning(insight)
                    else: st.error(insight)
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาด: {e}")
 
# PAGE: NETFLIX PREDICT

elif page == "🎬  Netflix Predict":
    st.markdown('<div class="page-title">Netflix Popularity Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">กรอกข้อมูล Content เพื่อทำนายโอกาสได้รับความนิยมบน Netflix พร้อมรับคำแนะนำ</div>', unsafe_allow_html=True)
 
    st.markdown("### 📝 Input Features")
    st.markdown('<p class="tooltip-label" style="margin-bottom:20px;">กำหนดรูปแบบและรายละเอียดของคอนเทนต์ที่ต้องการวิเคราะห์</p>', unsafe_allow_html=True)
 
    col1, col2, col3 = st.columns(3)
    with col1:
        t = st.selectbox(
            "🎬 Content Type",
            options=["Movie", "TV Show"],
            help="Movie = ภาพยนตร์ | TV Show = ซีรีส์/รายการ"
        )
        st.markdown('<p class="tooltip-label">💡 TV Show มักสร้าง Watch Time ต่อ User สูงกว่า</p>', unsafe_allow_html=True)
 
    with col2:
        year = st.number_input(
            "📅 Release Year",
            min_value=1980, max_value=2030, value=2024, step=1,
            help="ปีที่ Content ออกฉาย — Content ใหม่กว่ามักได้รับ Promotion มากกว่า"
        )
        st.markdown('<p class="tooltip-label">💡 โมเดลให้น้ำหนักคอนเทนต์ใหม่สูงกว่าคอนเทนต์เก่า</p>', unsafe_allow_html=True)
 
    with col3:
        duration = st.number_input(
            "⏱ Duration (minutes)",
            min_value=1, max_value=300, value=90, step=5,
            help="สำหรับ Movie = ความยาวรวม | TV Show = นาทีต่อ Episode"
        )
        st.markdown('<p class="tooltip-label">💡 ระวังความยาวที่มากเกินไปอาจทำให้ Completion rate ตก</p>', unsafe_allow_html=True)
 
    # [เพิ่ม] ส่วนให้ User เลือก Genre
    st.markdown("<br>", unsafe_allow_html=True)
    selected_genres = []
    if netflix_genre_columns:
        selected_genres = st.multiselect(
            "🎭 Genres (listed_in)",
            options=netflix_genre_columns,
            help="เลือกหมวดหมู่ของคอนเทนต์ (สามารถเลือกได้มากกว่า 1 หมวดหมู่)"
        )
        st.markdown('<p class="tooltip-label">💡 หมวดหมู่ที่ตรงกับกลุ่มเป้าหมายมีผลอย่างมากต่อความนิยม</p>', unsafe_allow_html=True)
    else:
        st.info("⚠️ ไม่พบข้อมูล Genre กรุณารันไฟล์ train_netflix.py เพื่อฝึกโมเดลและสร้างไฟล์ genre_columns.pkl ก่อน")

    st.markdown("<br>", unsafe_allow_html=True)
 
    # [แก้ไข] เช็คว่าโหลด genre_columns สำเร็จด้วย
    models_ready_n = netflix_model and netflix_scaler and netflix_le and netflix_genre_columns
    col_ml2, col_nn2 = st.columns(2)
 
    with col_ml2:
        if st.button("🧠 Predict with ML Ensemble", use_container_width=True, key="nf_ml"):
            if not models_ready_n:
                st.warning("⚠️ ไม่พบไฟล์โมเดล ML หรือไฟล์ Genre — กรุณาฝึกโมเดลก่อน (models/netflix/model.pkl)")
            else:
                try:
                    t_enc = netflix_le.transform([t])[0]
                    # [เพิ่ม] แปลง Genre ที่เลือกให้เป็น One-Hot Array
                    genre_features = [1 if g in selected_genres else 0 for g in netflix_genre_columns]
                    # รวม Features ทั้งหมดเข้าด้วยกัน (เรียงลำดับให้ตรงกับตอน Train)
                    features = [t_enc, year, duration] + genre_features
                    data = netflix_scaler.transform(np.array([features]))
                    
                    prob = netflix_model.predict_proba(data)[0][1]
                    verdict, insight, theme = get_actionable_insight(prob)
                    
                    st.markdown(f'''<div class="result-box" style="border-color: rgba(247,95,142,0.3);">
<div class="result-percent" style="background: linear-gradient(to right, var(--accent-pink), #ffb3c8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{prob*100:.1f}%</div>
<div class="result-label">ML Ensemble Confidence Score</div>
<br><span style="font-size:1.3rem; font-weight:700;">{verdict}</span>
</div>''', unsafe_allow_html=True)
                    st.progress(float(prob))
                    
                    if theme == "success": st.success(insight)
                    elif theme == "warning": st.warning(insight)
                    else: st.error(insight)
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาด: {e}")
 
    with col_nn2:
        if st.button("🤖 Predict with Neural Net", use_container_width=True, key="nf_nn"):
            if not (models_ready_n and netflix_nn):
                st.warning("⚠️ ไม่พบไฟล์ Neural Network หรือไฟล์ Genre — กรุณาฝึกโมเดลก่อน (models/netflix/nn.keras)")
            else:
                try:
                    t_enc = netflix_le.transform([t])[0]
                    # [เพิ่ม] แปลง Genre ที่เลือกให้เป็น One-Hot Array
                    genre_features = [1 if g in selected_genres else 0 for g in netflix_genre_columns]
                    # รวม Features ทั้งหมดเข้าด้วยกัน (เรียงลำดับให้ตรงกับตอน Train)
                    features = [t_enc, year, duration] + genre_features
                    data = netflix_scaler.transform(np.array([features]))
                    
                    prob = float(netflix_nn.predict(data, verbose=0)[0][0])
                    verdict, insight, theme = get_actionable_insight(prob)
                    
                    st.markdown(f'''<div class="result-box" style="border-color: rgba(247,95,142,0.3);">
<div class="result-percent" style="background: linear-gradient(to right, var(--accent-pink), var(--accent-blue)); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{prob*100:.1f}%</div>
<div class="result-label">Neural Network Confidence Score</div>
<br><span style="font-size:1.3rem; font-weight:700;">{verdict}</span>
</div>''', unsafe_allow_html=True)
                    st.progress(float(prob))
                    
                    if theme == "success": st.success(insight)
                    elif theme == "warning": st.warning(insight)
                    else: st.error(insight)
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาด: {e}")