
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

# Configuration de la page
st.set_page_config(
    page_title="Prédilo AI - Expertise Immobilière",
    page_icon="💎",
    layout="wide"
)

# Style CSS Premium personnalisé (Glassmorphism & Gradients)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    /* Background général */
    .stApp {
        background: radial-gradient(circle at top left, #0e1117, #1c2331);
        color: #e0e0e0;
    }

    /* Boîtes de prédiction Glassmorphism */
    .premium-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 25px;
    }

    /* Métriques personnalisées */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        color: #4facfe !important;
    }
    
    /* Titres avec dégradé */
    .title-gradient {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 30px;
    }

    /* Tabs stylisées */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(255,255,255,0.05);
        border-radius: 10px;
        color: white;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# 1. LE JEU DE DONNÉES (Dataset - 30 exemples exacts)
# ==========================================================
@st.cache_data
def load_data():
    donnees = {
        'm2':      [30, 50, 70, 90, 110, 130, 30, 50, 70, 50, 45, 85, 120, 60, 95, 55, 75, 100, 65, 80, 40, 90, 115, 140, 35, 60, 95, 105, 125, 150],
        'dist':    [1, 2, 3, 5, 8, 10, 15, 20, 25, 2, 4, 6, 12, 5, 8, 3, 4, 7, 2, 5, 1, 6, 9, 11, 18, 3, 7, 8, 10, 13],
        'neuf':    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1],
        'pieces':  [1, 2, 3, 4, 4, 5, 1, 2, 3, 2, 2, 3, 4, 3, 4, 2, 3, 4, 3, 3, 1, 4, 5, 6, 1, 2, 4, 4, 5, 6],
        'parking': [0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1],
        'etage':   [0, 2, 1, 3, 0, 1, 0, 1, 0, 4, 2, 3, 0, 2, 5, 3, 1, 2, 5, 4, 0, 3, 1, 0, 2, 4, 6, 2, 1, 3],
        'balcon':  [0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1],
        'jardin':  [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1],
        'ascenseur':[0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1],
        'dpe':     [4, 3, 4, 3, 5, 4, 6, 5, 6, 2, 4, 3, 1, 4, 3, 2, 4, 2, 1, 3, 5, 2, 3, 4, 6, 2, 3, 1, 3, 1],
        'annee':   [1985, 2000, 1995, 2005, 1980, 1990, 1975, 1988, 1970, 2020, 1998, 2008, 2022, 2002, 2010, 2018, 1992, 2015, 2021, 2005, 1982, 2012, 2003, 1995, 1978, 2019, 2011, 2023, 2007, 2024],
        'etat':    [2, 2, 2, 2, 1, 2, 1, 1, 1, 3, 2, 2, 3, 2, 2, 3, 2, 3, 3, 2, 1, 3, 2, 2, 1, 3, 2, 3, 2, 3],
        'prix':    [180000, 275000, 362000, 418000, 485000, 515000, 115000, 182000, 238000, 315000, 220000, 380000, 540000, 290000, 430000, 298000, 355000, 475000, 325000, 395000, 195000, 445000, 520000, 580000, 145000, 310000, 460000, 495000, 535000, 625000]
    }
    return pd.DataFrame(donnees)

df = load_data()

# ==========================================================
# 2. ENTRAÎNEMENT DE L'IA (Train / Test Split)
# ==========================================================
features = ['m2', 'dist', 'neuf', 'pieces', 'parking', 'etage', 'balcon', 'jardin', 'ascenseur', 'dpe', 'annee', 'etat']
X = df[features]
y = df['prix']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred_test = model.predict(X_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
score_r2 = r2_score(y_test, y_pred_test)

# ==========================================================
# 3. INTERFACE PRINCIPALE
# ==========================================================
st.markdown('<h1 class="title-gradient">💎 Prédilo AI Premium</h1>', unsafe_allow_html=True)

# Barre latérale luxueuse
with st.sidebar:
    st.markdown("### 🏗️ Configuration de l'Expertise")
    surface = st.slider("Surface habitable (m2)", 20, 300, 85, help="Surface totale en mètres carrés")
    pieces = st.select_slider("Nombre de pièces", options=[1, 2, 3, 4, 5, 6, 7, 8], value=4)
    dist = st.slider("Distance du centre-ville (km)", 0, 50, 5)
    annee = st.number_input("Année de construction", 1900, 2025, 2012)
    
    st.markdown("---")
    st.markdown("### ✨ Caractéristiques")
    col_a, col_b = st.columns(2)
    with col_a:
        neuf = st.checkbox("Neuf", value=False)
        parking = st.checkbox("Parking", value=True)
    with col_b:
        balcon = st.checkbox("Balcon", value=True)
        jardin = st.checkbox("Jardin", value=False)
    
    st.markdown("---")
    etat_label = st.radio("État général", ["À rénover", "Correct", "Excellent"], horizontal=True)
    dpe_label = st.select_slider("Performance énergétique (DPE)", options=["A", "B", "C", "D", "E", "F", "G"], value="C")
    
    st.markdown("---")
    prix_demande = st.number_input("Prix demandé par le vendeur (€)", 50000, 1500000, 420000)

# Conversion des données
code_neuf = 1 if neuf else 0
code_park = 1 if parking else 0
code_balc = 1 if balcon else 0
code_jard = 1 if jardin else 0
code_etat = {"À rénover": 1, "Correct": 2, "Excellent": 3}[etat_label]
code_dpe = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}[dpe_label]

# ==========================================================
# 4. ANALYSE ET PRÉDICTION
# ==========================================================
# Calcul de l'estimation
X_user = [[surface, dist, code_neuf, pieces, code_park, 0, code_balc, code_jard, 1, code_dpe, annee, code_etat]]
prediction = model.predict(X_user)[0]

# Résumé des métriques
cols = st.columns([2, 1, 1, 1])
with cols[0]:
    st.markdown(f"""
    <div class="premium-card">
        <p style="margin:0; font-size:14px; opacity:0.8;">ESTIMATION IA PRÉDILO</p>
        <h2 style="margin:0; font-size:50px;">{prediction:,.0f} €</h2>
    </div>
    """, unsafe_allow_html=True)
with cols[1]:
    st.metric("Prix Vendeur", f"{prix_demande:,.0f} €", delta=f"{prediction - prix_demande:,.0f} €", delta_color="normal")
with cols[2]:
    st.metric("Précision (Examen)", f"{score_r2:.1%}")
with cols[3]:
    st.metric("Fiabilité Réelle", f"± {mae_test:,.0f} €")

# Verdict intelligent
st.markdown('<div class="premium-card">', unsafe_allow_html=True)
diff_perc = ((prix_demande - prediction) / prediction) * 100
if prix_demande < (prediction - mae_test):
    st.success(f"🎯 **RECOMMANDATION : EXCELLENTE AFFAIRE !** Le bien est sous-évalué de {abs(diff_perc):.1f}% par rapport au marché.")
elif prix_demande > (prediction + mae_test):
    st.error(f"❌ **RECOMMANDATION : TROP CHER !** Le prix demandé est {diff_perc:.1f}% plus élevé que l'estimation de l'IA.")
else:
    st.info("⚖️ **RECOMMANDATION : PRIX JUSTE.** Le tarif est conforme à la valeur intrinsèque du bien.")
st.markdown('</div>', unsafe_allow_html=True)

# ==========================================================
# 5. VISUALISATIONS HAUT DE GAMME
# ==========================================================
tabs = st.tabs(["📊 Analyse de Marché", "🧪 Pensée de l'IA", "📄 Rapport d'Expert"])

with tabs[0]:
    c1, c2 = st.columns(2)
    with c1:
        # Gauge Chart pour l'attractivité
        score_attr = max(10, min(100, 100 - diff_perc))
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = score_attr,
            title = {'text': "Indice d'Attractivité (%)"},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "#4facfe"},
                'steps': [
                    {'range': [0, 40], 'color': "rgba(255,0,0,0.1)"},
                    {'range': [40, 70], 'color': "rgba(255,165,0,0.1)"},
                    {'range': [70, 100], 'color': "rgba(0,128,0,0.1)"}],
            }
        ))
        fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with c2:
        # Histogramme stylisé
        fig_hist = px.histogram(df, x="prix", title="Où se situe ce bien dans le secteur ?", 
                               color_discrete_sequence=['#4facfe'], opacity=0.7)
        fig_hist.add_vline(x=prediction, line_dash="dash", line_color="#00f2fe", annotation_text="VOTRE BIEN")
        fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
        st.plotly_chart(fig_hist, use_container_width=True)

with tabs[1]:
    st.write("### Comment l'IA a-t-elle calculé ce prix ?")
    # Graphique en cascade simplifié (Waterfall)
    base_price = 150000 # On estime un prix de base
    importances = model.coef_
    
    fig_water = go.Figure(go.Waterfall(
        name = "20", orientation = "v",
        measure = ["relative", "relative", "relative", "relative", "total"],
        x = ["Prix de base", "Effet Surface", "Effet Secteur", "Effet Bonus État", "Estimation Finale"],
        textposition = "outside",
        text = [f"{base_price:,.0f}€", f"+{importances[0]*surface:,.0f}€", f"{importances[1]*dist:,.0f}€", f"+{importances[11]*code_etat:,.0f}€", f"{prediction:,.0f}€"],
        y = [base_price, importances[0]*surface, importances[1]*dist, importances[11]*code_etat, 0],
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
    ))
    fig_water.update_layout(title = "Décomposition de la valeur", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
    st.plotly_chart(fig_water, use_container_width=True)
    st.info("💡 L'IA analyse les 12 caractéristiques simultanément pour équilibrer chaque détail.")

with tabs[2]:
    st.markdown(f"""
    ### 📝 Résumé de l'Expertise
    
    Le bien situé à **{dist} km** du centre, d'une surface de **{surface} m²**, présente un score de précision de **{score_r2:.1%}**.
    
    **Points Forts :**
    - État général : **{etat_label}**
    - Énergie : **Classe {dpe_label}**
    - Espaces : **{pieces} pièces** {'avec balcon' if balcon else 'sans balcon'}.
    
    **Analyse Financière :**
    L'IA Prédilo estime que la valeur de marché est de **{prediction:,.0f} €**. Le vendeur demande **{prix_demande:,.0f} €**. 
    L'investissement est donc considéré comme **{('favorable' if diff_perc < 0 else 'à négocier')}**.
    """)

st.markdown("---")
st.markdown("<center style='opacity:0.5;'>Propulsé par la Force de l'IA Prédilo Premium 💎</center>", unsafe_allow_html=True)


