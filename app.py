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
    page_title="IA Prédilo - Immobilier",
    page_icon="🏠",
    layout="wide"
)

# Style CSS pour une interface premium
st.markdown("""
    <style>
    .prediction-box {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 15px;
        border-left: 10px solid #007bff;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 20px 0;
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
# 3. INTERFACE (Sidebar & Menu)
# ==========================================================
st.sidebar.header("🔧 Paramètres du bien")

surface = st.sidebar.slider("Surface (m2)", 20, 300, 75)
pieces = st.sidebar.number_input("Nombre de pièces", 1, 10, 3)
distance = st.sidebar.slider("Distance centre (km)", 0, 50, 5)
neuf_o_n = st.sidebar.selectbox("Est-ce neuf ?", ["NON", "OUI"])
annee = st.sidebar.number_input("Année de construction", 1900, 2025, 2010)
etat = st.sidebar.select_slider("État général", options=["A rénover", "Bon", "Excellent"], value="Bon")
dpe_label = st.sidebar.select_slider("Classe DPE", options=["A", "B", "C", "D", "E", "F", "G"], value="C")

st.sidebar.subheader("Équipements")
parking = st.sidebar.checkbox("Parking / Garage", value=True)
balcon = st.sidebar.checkbox("Balcon / Terrasse")
jardin = st.sidebar.checkbox("Jardin")
ascenseur = st.sidebar.checkbox("Ascenseur")

prix_vendeur = st.sidebar.number_input("Prix du vendeur (€)", 50000, 1000000, 350000)

# Conversion des entrées pour l'IA
code_neuf = 1 if neuf_o_n == "OUI" else 0
code_etat = {"A rénover": 1, "Bon": 2, "Excellent": 3}[etat]
code_dpe = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}[dpe_label]
code_park = 1 if parking else 0
code_balc = 1 if balcon else 0
code_jard = 1 if jardin else 0
code_asc = 1 if ascenseur else 0

# ==========================================================
# 4. CALCULS ET RÉSULTATS
# ==========================================================
st.title("🚀 Prédilo IA : Expert Immobilier")
st.markdown("---")

# Prédiction en temps réel
input_data = [[surface, distance, code_neuf, pieces, code_park, 0, code_balc, code_jard, code_asc, code_dpe, annee, code_etat]]
estimation = model.predict(input_data)[0]

# Affichage des métriques clés
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Estimation IA", f"{estimation:,.0f} €")
with c2: st.metric("Prix Vendeur", f"{prix_vendeur:,.0f} €")
with c3: st.metric("Précision IA", f"{score_r2:.1%}")
with c4: st.metric("Erreur (Test)", f"± {mae_test:,.0f} €")

# Verdict visuel
st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
if prix_vendeur < (estimation - mae_test):
    st.success("🎯 **VERDICT : EXCELLENTE AFFAIRE !**")
    st.write(f"Ce bien est sous-évalué d'environ **{estimation - prix_vendeur:,.0f} €**.")
elif prix_vendeur > (estimation + mae_test):
    st.error("❌ **VERDICT : TROP CHER !**")
    st.write(f"Le prix est supérieur à la réalité du marché de **{prix_vendeur - estimation:,.0f} €**.")
else:
    st.info("⚖️ **VERDICT : PRIX CORRECT !**")
    st.write("Le prix est parfaitement aligné avec les tendances actuelles.")
st.markdown('</div>', unsafe_allow_html=True)

# ==========================================================
# 5. GRAPHIQUES PLOTLY
# ==========================================================
t1, t2 = st.tabs(["📊 Distribution du Marché", "📈 Relation Surface/Prix"])

with t1:
    fig_hist = px.histogram(df, x="prix", title="Où se situe votre bien ?", color_discrete_sequence=['#007bff'])
    fig_hist.add_vline(x=estimation, line_dash="dash", line_color="red", annotation_text="VOTRE ESTIMATION")
    st.plotly_chart(fig_hist, use_container_width=True)

with t2:
    fig_scat = px.scatter(df, x="m2", y="prix", title="Prix en fonction de la Surface", labels={"m2": "Surface (m2)", "prix": "Prix (€)"})
    fig_scat.add_trace(go.Scatter(x=[surface], y=[estimation], mode='markers', name='Votre Bien', marker=dict(size=15, color='red', symbol='star')))
    st.plotly_chart(fig_scat, use_container_width=True)

st.divider()
st.caption("Développé avec ❤️ pour votre apprentissage de l'IA.")


