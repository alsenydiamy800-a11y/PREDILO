# MON PREMIER PROJET D'IA : PREDICTION IMMOBILIERE
# ==========================================================
# Ce programme est le resultat de 6 etapes d'apprentissage.
# Il utilise le Machine Learning pour analyser le marche.
# VERSION ENRICHIE avec 12 caracteristiques et 30 exemples

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================================
# 1. LE JEU DE DONNEES (LE "DATASET")
# ==========================================================
# C'est la base de connaissances de l'IA.
# Plus il y a d'exemples, plus l'IA sera precise !
# 
# LEGENDE DES COLONNES :
# - m2        : Surface en metres carres
# - dist      : Distance du centre-ville en km
# - neuf      : 1 = Construction neuve, 0 = Ancien
# - pieces    : Nombre de pieces (chambres + salon)
# - parking   : 1 = Avec parking/garage, 0 = Sans
# - etage     : Numero d'etage (0 = Rez-de-chaussee)
# - balcon    : 1 = Avec balcon/terrasse, 0 = Sans
# - jardin    : 1 = Avec jardin/espace vert, 0 = Sans
# - ascenseur : 1 = Avec ascenseur, 0 = Sans
# - dpe       : Diagnostic Performance Energetique (1=A excellent, 7=G mauvais)
# - annee     : Annee de construction
# - etat      : Etat general (1=A renover, 2=Bon, 3=Excellent)
# - prix      : Prix de vente en euros

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

print("=" * 70)
print("STATISTIQUES DU DATASET")
print("=" * 70)
print(f"Nombre total d'exemples : {len(df)}")
print(f"Nombre de caracteristiques : 12")
print(f"Prix moyen du marche    : {df['prix'].mean():,.0f} euros")
print(f"Prix minimum observe    : {df['prix'].min():,.0f} euros")
print(f"Prix maximum observe    : {df['prix'].max():,.0f} euros")
print(f"Surface moyenne         : {df['m2'].mean():.1f} m2")
print(f"Annee moyenne           : {df['annee'].mean():.0f}")
print("=" * 70)
print()

# ==========================================================
# 2. L'ENTRAINEMENT (LE "FITTING")
# ==========================================================
# Le modele observe les donnees et cherche une regle logique.
# Il analyse comment chaque caracteristique influence le prix.

X = df[['m2', 'dist', 'neuf', 'pieces', 'parking', 'etage', 'balcon', 'jardin', 'ascenseur', 'dpe', 'annee', 'etat']]
y = df['prix']

ia = LinearRegression()
ia.fit(X, y)

# Affichage de l'importance de chaque caracteristique
print("APPRENTISSAGE TERMINE - Influence de chaque critere :")
print("-" * 70)
caracteristiques = ['Surface (m2)', 'Distance (km)', 'Neuf', 'Pieces', 'Parking', 'Etage', 'Balcon', 'Jardin', 'Ascenseur', 'DPE', 'Annee', 'Etat']
for i, nom in enumerate(caracteristiques):
    coef = ia.coef_[i]
    signe = "+" if coef > 0 else "-"
    print(f"[{signe}] {nom:20s} : {coef:+12,.2f} euros par unite")
print(f"Prix de base             : {ia.intercept_:>12,.0f} euros")
print("-" * 70)
print()

# ==========================================================
# 3. LA FIABILITE (L'ERREUR MAE ET R2)
# ==========================================================
# On calcule de combien l'IA se trompe par rapport a la realite.
# MAE = Erreur moyenne en euros
# R2 = Score de precision (0 a 1, plus c'est proche de 1, mieux c'est)

predictions = ia.predict(X)
erreur = mean_absolute_error(y, predictions)
score_r2 = r2_score(y, predictions)

print("=" * 70)
print("EVALUATION DE LA PRECISION DU MODELE")
print("=" * 70)
print(f"Erreur moyenne (MAE)    : {erreur:,.0f} euros")
print(f"Score de precision (R2) : {score_r2:.2%}")
if score_r2 > 0.95:
    print(">> EXCELLENT ! Le modele est extremement fiable.")
elif score_r2 > 0.90:
    print(">> Tres bon ! Le modele est tres fiable.")
elif score_r2 > 0.75:
    print(">> Bon ! Le modele est assez fiable.")
else:
    print(">> Attention ! Le modele pourrait etre ameliore.")
print("=" * 70)
print()

# ==========================================================
# DEBUT DU PROGRAMME INTERACTIF
# ==========================================================
print("=" * 70)
print("SYSTEME D'ANALYSE IMMOBILIERE PAR IA - VERSION COMPLETE")
print("=" * 70)
print(f"Statut : Entrainement reussi sur {len(df)} exemples.")
print(f"Precision : Marge d'erreur moyenne de {erreur:,.0f} euros.")
print(f"Modele : 12 caracteristiques analysees")
print("=" * 70)
print()

try:
    # ==========================================================
    # 4. LA SAISIE UTILISATEUR
    # ==========================================================
    # L'utilisateur entre les caracteristiques de la maison a analyser.
    
    print("ENTREZ LES CARACTERISTIQUES DU BIEN :")
    print("-" * 70)
    
    s = float(input("Surface de la maison (m2) : "))
    p = int(input("Nombre de pieces : "))
    d = float(input("Distance du centre-ville (km) : "))
    n = input("Est-ce neuf ? (OUI/NON) : ").upper()
    park = input("Y a-t-il un parking ? (OUI/NON) : ").upper()
    etg = int(input("A quel etage ? (0 pour RDC) : "))
    balc = input("Y a-t-il un balcon ? (OUI/NON) : ").upper()
    jard = input("Y a-t-il un jardin ? (OUI/NON) : ").upper()
    asc = input("Y a-t-il un ascenseur ? (OUI/NON) : ").upper()
    dpe_input = int(input("DPE (1=A excellent, 7=G mauvais) : "))
    annee_input = int(input("Annee de construction (ex: 2010) : "))
    etat_input = int(input("Etat (1=A renover, 2=Bon, 3=Excellent) : "))
    prix_voulu = float(input("Prix affiche par le vendeur (euros) : "))
    
    # Conversion des reponses textuelles en codes numeriques
    code_neuf = 1 if n == "OUI" else 0
    parking = 1 if park == "OUI" else 0
    balcon = 1 if balc == "OUI" else 0
    jardin = 1 if jard == "OUI" else 0
    ascenseur = 1 if asc == "OUI" else 0
    
    print()
    print("Analyse en cours avec 12 caracteristiques...")
    print()
    
    # ==========================================================
    # 5. LA PREDICTION
    # ==========================================================
    # L'IA calcule le prix estime base sur ce qu'elle a appris.
    
    estimation = ia.predict([[s, d, code_neuf, p, parking, etg, balcon, jardin, ascenseur, dpe_input, annee_input, etat_input]])[0]
    
    # Calcul de la fourchette de prix acceptable
    prix_min = estimation - erreur
    prix_max = estimation + erreur
    
    print("=" * 70)
    print("RESULTAT DE L'ANALYSE")
    print("=" * 70)
    print(f"Estimation de l'IA  : {estimation:>15,.0f} euros")
    print(f"Prix vendeur        : {prix_voulu:>15,.0f} euros")
    print(f"Fourchette basse    : {prix_min:>15,.0f} euros")
    print(f"Fourchette haute    : {prix_max:>15,.0f} euros")
    
    # Calcul de l'ecart en pourcentage
    ecart = ((prix_voulu - estimation) / estimation) * 100
    print(f"Ecart               : {ecart:>14.1f} %")
    print("=" * 70)
    print()
    
    # ==========================================================
    # 6. LE VERDICT : EST-CE UNE BONNE AFFAIRE ?
    # ==========================================================
    # L'IA compare le prix demande avec son estimation et donne un conseil.
    
    print("VERDICT FINAL :")
    print("-" * 70)
    
    if prix_voulu < (estimation - erreur):
        economie = estimation - prix_voulu
        print(">> C'est une EXCELLENTE AFFAIRE !")
        print(f"Vous economisez environ {economie:,.0f} euros par rapport au marche.")
        print("RECOMMANDATION : Foncez, c'est une opportunite !")
        
    elif prix_voulu > (estimation + erreur):
        surpix = prix_voulu - estimation
        print(">> C'est TROP CHER pour ce que c'est.")
        print(f"Vous payez environ {surpix:,.0f} euros de plus que le prix du marche.")
        print("RECOMMANDATION : Negociez ou cherchez ailleurs.")
        
    else:
        print(">> C'est un PRIX CORRECT par rapport au marche.")
        print("Le prix demande est dans la fourchette acceptable.")
        print("RECOMMANDATION : Prix raisonnable, a vous de decider.")
    
    print("=" * 70)

except ValueError:
    print()
    print("ERREUR : Veuillez entrer des chiffres valides.")
    print("Conseil : Utilisez uniquement des nombres (ex: 75, 3, 2010, 250000)")

# ==========================================================
# RESUME DE VOTRE APPRENTISSAGE
# ==========================================================
"""
FELICITATIONS ! Vous avez appris les concepts cles de l'IA :

CONCEPTS FONDAMENTAUX :
- DATASET    : Les exemples que vous donnez pour apprendre (30 maisons)
- FEATURES   : Les criteres que l'IA analyse (12 caracteristiques)
- TARGET     : Ce que l'IA doit predire (le prix)
- MODEL      : Le cerveau mathematique (Linear Regression)
- FIT        : L'action d'apprendre a partir des donnees
- PREDICT    : L'action de deviner sur un nouvel exemple
- MAE        : La marge d'erreur moyenne en euros
- R2         : Le score de precision du modele (0 a 1)

LES 12 CARACTERISTIQUES ANALYSEES :
1.  Surface (m2)        - Plus c'est grand, plus c'est cher
2.  Distance (km)       - Plus c'est loin du centre, moins c'est cher
3.  Neuf/Ancien         - Le neuf coute generalement plus cher
4.  Nombre de pieces    - Plus de pieces = prix plus eleve
5.  Parking             - Ajoute de la valeur (10-20k euros)
6.  Etage               - Les etages eleves sont souvent plus chers
7.  Balcon              - Tres recherche, augmente le prix
8.  Jardin              - Tres valorise, surtout en ville (+20-30k euros)
9.  Ascenseur           - Important pour les etages eleves (+10-15k euros)
10. DPE                 - Performance energetique (A=1 meilleur, G=7 pire)
11. Annee               - L'age du bien influence le prix
12. Etat                - Travaux a prevoir ou non (1=Renover, 3=Excellent)

POUR ALLER PLUS LOIN :
- Ajoutez plus d'exemples dans le dataset (50, 100, 1000...)
- Testez d'autres modeles (Random Forest, XGBoost...)
- Ajoutez de nouvelles caracteristiques (cave, garage double, piscine...)
- Creez une interface graphique avec Tkinter ou Streamlit
- Visualisez les donnees avec Matplotlib ou Seaborn
"""
