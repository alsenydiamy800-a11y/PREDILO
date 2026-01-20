import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Analyse Immobilière IA", layout="wide")

# ========== TITRE ==========
st.title("🏠 Système d'Analyse Immobilière par IA")
st.markdown("---")

# ========== DATASET ==========
donnees = {
    'm2':      [30,  50,  70,  90,  110, 130, 30,  50,  70,  50,  45,  85,  120, 60,  95,  
                55,  75,  100, 65,  80,  40,  90,  115, 140, 35,  60,  95,  105, 125, 150],
    
    'dist':    [1,   2,   3,   5,   8,   10,  15,  20,  25,  2,   4,   6,   12,  5,   8,   
                3,   4,   7,   2,   5,   1,   6,   9,   11,  18,  3,   7,   8,   10,  13],
    
    'neuf':    [0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0,   1,   0,   0,   
                1,   0,   0,   1,   0,   0,   1,   0,   0,   0,   1,   0,   1,   0,   1],
    
    'pieces':  [1,   2,   3,   4,   4,   5,   1,   2,   3,   2,   2,   3,   4,   3,   4,   
                2,   3,   4,   3,   3,   1,   4,   5,   6,   1,   2,   4,   4,   5,   6],
    
    'parking': [0,   1,   0,   1,   1,   1,   0,   0,   0,   1,   1,   1,   1,   0,   1,   
                1,   0,   1,   1,   1,   0,   1,   1,   1,   0,   1,   1,   1,   1,   1],
    
    'etage':   [0,   2,   1,   3,   0,   1,   0,   1,   0,   4,   2,   3,   0,   2,   5,   
                3,   1,   2,   5,   4,   0,   3,   1,   0,   2,   4,   6,   2,   1,   3],
    
    'balcon':  [0,   1,   0,   1,   0,   1,   0,   0,   0,   1,   1,   1,   1,   1,   1,   
                1,   0,   1,   1,   1,   0,   1,   1,   1,   0,   1,   1,   1,   1,   1],
    
    'jardin':  [0,   0,   0,   0,   1,   1,   0,   0,   0,   0,   0,   1,   1,   0,   1,   
                0,   0,   1,   0,   0,   0,   1,   1,   1,   0,   0,   1,   1,   1,   1],
    
    'ascenseur': [0,   1,   1,   1,   0,   1,   0,   0,   0,   1,   1,   1,   0,   1,   1,   
                  1,   1,   1,   1,   1,   0,   1,   1,   0,   0,   1,   1,   1,   1,   1],
    
    'dpe':     [4,   3,   4,   3,   5,   4,   6,   5,   6,   2,   4,   3,   1,   4,   3,   
                2,   4,   2,   1,   3,   5,   2,   3,   4,   6,   2,   3,   1,   3,   1],
    
    'annee':   [1985, 2000, 1995, 2005, 1980, 1990, 1975, 1988, 1970, 2020, 1998, 2008, 2022, 2002, 2010,
                2018, 1992, 2015, 2021, 2005, 1982, 2012, 2003, 1995, 1978, 2019, 2011, 2023, 2007, 2024],
    
    'etat':    [2,   2,   2,   2,   1,   2,   1,   1,   1,   3,   2,   2,   3,   2,   2,   
                3,   2,   3,   3,   2,   1,   3,   2,   2,   1,   3,   2,   3,   2,   3],
    
    'prix':    [180000, 275000, 362000, 418000, 485000, 515000, 115000, 182000, 238000, 315000, 
                220000, 380000, 540000, 290000, 430000, 298000, 355000, 475000, 325000, 395000,
                195000, 445000, 520000, 580000, 145000, 310000, 460000, 495000, 535000, 625000]
}

df = pd.DataFrame(donnees)

# ========== STATISTIQUES ==========
with st.expander("📊 Statistiques du Dataset"):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nombre d'exemples", len(df))
    col2.metric("Prix moyen", f"{df['prix'].mean():,.0f} €")
    col3.metric("Prix min", f"{df['prix'].min():,.0f} €")
    col4.metric("Prix max", f"{df['prix'].max():,.0f} €")

# ========== ENTRAÎNEMENT ==========
X = df[['m2', 'dist', 'neuf', 'pieces', 'parking', 'etage', 'balcon', 'jardin', 'ascenseur', 'dpe', 'annee', 'etat']]
y = df['prix']

ia = LinearRegression()
ia.fit(X, y)

predictions = ia.predict(X)
erreur = mean_absolute_error(y, predictions)
score_r2 = r2_score(y, predictions)

# ========== AFFICHAGE FIABILITÉ ==========
with st.expander("✅ Fiabilité du Modèle"):
    col1, col2 = st.columns(2)
    col1.metric("Erreur moyenne (MAE)", f"{erreur:,.0f} €")
    col2.metric("Précision (R²)", f"{score_r2:.1%}")
    
    if score_r2 > 0.95:
        st.success("✨ EXCELLENT ! Le modèle est extrêmement fiable.")
    elif score_r2 > 0.90:
        st.success("✅ Très bon ! Le modèle est très fiable.")
    elif score_r2 > 0.75:
        st.info("ℹ️ Bon ! Le modèle est assez fiable.")
    else:
        st.warning("⚠️ Le modèle pourrait être amélioré.")

st.markdown("---")

# ========== FORMULAIRE ==========
st.header("🏘️ Analysez un bien immobilier")

col1, col2 = st.columns(2)

with col1:
    s = st.number_input("Surface (m²)", min_value=10, max_value=500, value=75)
    d = st.number_input("Distance du centre (km)", min_value=0.0, max_value=30.0, value=5.0)
    p = st.number_input("Nombre de pièces", min_value=1, max_value=10, value=3)
    annee_input = st.number_input("Année de construction", min_value=1900, max_value=2025, value=2010)

with col2:
    n = st.selectbox("Neuf ?", ["OUI", "NON"])
    park = st.selectbox("Parking ?", ["OUI", "NON"])
    balc = st.selectbox("Balcon ?", ["OUI", "NON"])
    jard = st.selectbox("Jardin ?", ["OUI", "NON"])

col3, col4 = st.columns(2)

with col3:
    asc = st.selectbox("Ascenseur ?", ["OUI", "NON"])
    etg = st.number_input("Étage (0 = RDC)", min_value=0, max_value=20, value=0)
    dpe_input = st.selectbox("DPE (1=Excellent, 7=Mauvais)", [1, 2, 3, 4, 5, 6, 7], index=3)

with col4:
    etat_input = st.selectbox("État du bien", [1, 2, 3], format_func=lambda x: ["À rénover", "Bon", "Excellent"][x-1])
    prix_voulu = st.number_input("Prix affiché (€)", min_value=50000, max_value=1000000, value=300000)

# ========== PREDICTION ==========
if st.button("🔮 Analyser le bien", use_container_width=True):
    code_neuf = 1 if n == "OUI" else 0
    parking = 1 if park == "OUI" else 0
    balcon = 1 if balc == "OUI" else 0
    jardin = 1 if jard == "OUI" else 0
    ascenseur = 1 if asc == "OUI" else 0
    
    estimation = ia.predict([[s, d, code_neuf, p, parking, etg, balcon, jardin, ascenseur, dpe_input, annee_input, etat_input]])[0]
    
    prix_min = estimation - erreur
    prix_max = estimation + erreur
    ecart = ((prix_voulu - estimation) / estimation) * 100
    
    st.markdown("---")
    st.header("📋 Résultat de l'Analyse")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Estimation IA", f"{estimation:,.0f} €", f"{ecart:+.1f}%")
    col2.metric("Prix vendeur", f"{prix_voulu:,.0f} €")
    col3.metric("Marge d'erreur", f"±{erreur:,.0f} €")
    
    st.markdown("---")
    
    if prix_voulu < (estimation - erreur):
        economie = estimation - prix_voulu
        st.success(f"🎉 C'est une EXCELLENTE AFFAIRE !")
        st.write(f"Vous économisez environ **{economie:,.0f} €** par rapport au marché.")
        st.write("**Recommandation** : Foncez, c'est une opportunité !")
        
    elif prix_voulu > (estimation + erreur):
        surprix = prix_voulu - estimation
        st.error(f"❌ C'est TROP CHER pour ce bien !")
        st.write(f"Vous payez environ **{surprix:,.0f} €** de plus que le prix du marché.")
        st.write("**Recommandation** : Négociez ou cherchez ailleurs.")
        
    else:
        st.info("✅ C'est un PRIX CORRECT par rapport au marché")
        st.write("Le prix demandé est dans la fourchette acceptable.")
        st.write("**Recommandation** : Prix raisonnable, à vous de décider.")
