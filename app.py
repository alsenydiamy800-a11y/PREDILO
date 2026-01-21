import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Configuration de la page
st.set_page_config(page_title="Analyse Immobilière par IA", layout="wide")

# Titre
st.title("🏠 Analyse Immobilière par IA")
st.markdown("Prédiction de prix immobilier avec Machine Learning")

# DATASET
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
                1,   0,   1,   1,   1,   0,   1,   1,   1,   0,   1,   1,   1,   1],
    
    'etage':   [0,   2,   1,   3,   0,   1,   0,   1,   0,   4,   2,   3,   0,   2,   5,   
                3,   1,   2,   5,   4,   0,   3,   1,   0,   2,   4,   6,   2,   1,   3],
    
    'balcon':  [0,   1,   0,   1,   0,   1,   0,   0,   0,   1,   1,   1,   1,   1,   1,   
                1,   0,   1,   1,   1,   0,   1,   1,   1,   0,   1,   1,   1,   1],
    
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

# Entrainement du modèle
X = df[['m2', 'dist', 'neuf', 'pieces', 'parking', 'etage', 'balcon', 'jardin', 'ascenseur', 'dpe', 'annee', 'etat']]
y = df['prix']

ia = LinearRegression()
ia.fit(X, y)

predictions = ia.predict(X)
erreur = mean_absolute_error(y, predictions)
score_r2 = r2_score(y, predictions)

# =========================================================
# INTERFACE STREAMLIT
# =========================================================
st.title("🏠 Analyse Immobilière par IA")
st.write("Prédisez le prix de vente optimal pour un bien immobilier")

# Afficher les statistiques
col1, col2, col3, col4 = st.columns(4)
col1.metric("Prix moyen", f"{df['prix'].mean():,.0f} €")
col2.metric("Erreur MAE", f"±{erreur:,.0f} €")
col3.metric("Précision R²", f"{score_r2:.1%}")
col4.metric("Exemples", len(df))

st.divider()

# Saisie utilisateur
st.subheader("📋 Caractéristiques du bien")

col1, col2, col3 = st.columns(3)

with col1:
    surface = st.number_input("Surface (m²)", min_value=20, max_value=200, value=75)
    pieces = st.number_input("Nombre de pièces", min_value=1, max_value=6, value=3)
    distance = st.number_input("Distance du centre (km)", min_value=0, max_value=30, value=5)

with col2:
    neuf = st.selectbox("Construction neuve ?", ["Non", "Oui"])
    parking = st.selectbox("Parking/Garage ?", ["Non", "Oui"])
    balcon = st.selectbox("Balcon/Terrasse ?", ["Non", "Oui"])

with col3:
    jardin = st.selectbox("Jardin ?", ["Non", "Oui"])
    ascenseur = st.selectbox("Ascenseur ?", ["Non", "Oui"])
    etage = st.number_input("Étage (0=RDC)", min_value=0, max_value=10, value=2)

col1, col2, col3 = st.columns(3)

with col1:
    dpe = st.slider("DPE (1=A excellent, 7=G mauvais)", 1, 7, 3)
    
with col2:
    annee = st.number_input("Année de construction", min_value=1950, max_value=2025, value=2000)
    
with col3:
    etat = st.slider("État (1=À rénover, 3=Excellent)", 1, 3, 2)

prix_vendeur = st.number_input("Prix affiché (€)", min_value=50000, max_value=1000000, value=350000, step=10000)

st.divider()

# Bouton d'analyse
if st.button("🔍 Analyser le bien", use_container_width=True):
    
    # Conversion des réponses
    code_neuf = 1 if neuf == "Oui" else 0
    code_parking = 1 if parking == "Oui" else 0
    code_balcon = 1 if balcon == "Oui" else 0
    code_jardin = 1 if jardin == "Oui" else 0
    code_ascenseur = 1 if ascenseur == "Oui" else 0
    
    # Prédiction
    estimation = ia.predict([[surface, distance, code_neuf, pieces, code_parking, etage, code_balcon, code_jardin, code_ascenseur, dpe, annee, etat]])[0]
    
    prix_min = estimation - erreur
    prix_max = estimation + erreur
    ecart = ((prix_vendeur - estimation) / estimation) * 100
    
    # Affichage des résultats
    st.subheader("📊 Résultat de l'analyse")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Estimation IA", f"{estimation:,.0f} €")
    col2.metric("Prix vendeur", f"{prix_vendeur:,.0f} €")
    col3.metric("Fourchette basse", f"{prix_min:,.0f} €")
    col4.metric("Fourchette haute", f"{prix_max:,.0f} €")
    
    st.metric("Écart", f"{ecart:+.1f}%")
    
    st.divider()
    
    # Verdict
    st.subheader("🎯 Verdict")
    
    if prix_vendeur < (estimation - erreur):
        economie = estimation - prix_vendeur
        st.success(f"✅ **EXCELLENTE AFFAIRE !**")
        st.write(f"Vous économisez environ **{economie:,.0f} €** par rapport au marché.")
        st.write("**Recommandation : Foncez, c'est une opportunité !**")
        
    elif prix_vendeur > (estimation + erreur):
        surprix = prix_vendeur - estimation
        st.warning(f"⚠️ **C'est TROP CHER pour ce que c'est.**")
        st.write(f"Vous payez environ **{surprix:,.0f} €** de plus que le prix du marché.")
        st.write("**Recommandation : Négociez ou cherchez ailleurs.**")
        
    else:
        st.info(f"ℹ️ **C'est un PRIX CORRECT par rapport au marché.**")
        st.write("Le prix demandé est dans la fourchette acceptable.")
        st.write("**Recommandation : Prix raisonnable, à vous de décider.**")

st.divider()

# Information supplémentaire
with st.expander("ℹ️ À propos du modèle"):
    st.write("**Modèle :** Régression linéaire avec 12 caractéristiques")
    st.write(f"**Données d'entraînement :** {len(df)} exemples")
    st.write(f"**Précision (R²) :** {score_r2:.2%}")
    st.write(f"**Marge d'erreur moyenne :** ±{erreur:,.0f} €")
    
    st.write("**Les 12 caractéristiques analysées :**")
    caracteristiques = ['Surface (m²)', 'Distance (km)', 'Neuf', 'Pièces', 'Parking', 'Étage', 
                       'Balcon', 'Jardin', 'Ascenseur', 'DPE', 'Année', 'État']
    for i, nom in enumerate(caracteristiques):
        coef = ia.coef_[i]
        signe = "+" if coef > 0 else "-"
        st.write(f"[{signe}] **{nom}** : {coef:+,.0f} € par unité")
