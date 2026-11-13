# ══════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    silhouette_score
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

sns.set_theme(style="whitegrid")

# ══════════════════════════════════════════════════════════════
# CHARGEMENT DES DONNÉES
# ══════════════════════════════════════════════════════════════
import pandas as pd

df = pd.read_csv("data/Walmart_customer_purchases.csv")

print("DF chargé :", df.shape)
# Groupe d'âge
bins = [18, 25, 35, 45, 55, 60]
labels = ['18-25', '26-35', '36-45', '46-55', '56-60']
df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True)

print("\n" + "=" * 60)
print("CHARGEMENT DES DONNÉES")
print("=" * 60)
print(f"Dimensions initiales : {df.shape}")
print("\nAperçu des données :")
print(df.head().to_string())

# ══════════════════════════════════════════════════════════════
# NETTOYAGE DES DONNÉES
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("NETTOYAGE DES DONNÉES")
print("=" * 60)

print("\nValeurs manquantes par colonne :")
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
print(pd.DataFrame({
    'Manquantes': missing,
    'Pourcentage (%)': missing_pct
}).to_string())

avant = len(df)
df = df.dropna().copy()
print(f"\nLignes supprimées (NA) : {avant - len(df)}")
print(f"Lignes restantes       : {len(df)}")

print("\nVérification des doublons :")
print(f"Lignes dupliquées : {df.duplicated().sum()}")
if 'Customer_ID' in df.columns:
    print(f"Customer_ID dupliqués : {df.duplicated(subset=['Customer_ID']).sum()}")

df = df.drop_duplicates().copy()
print(f"Lignes restantes après suppression des doublons : {len(df)}")

print("\nDétection des valeurs aberrantes (IQR) :")

def detecter_aberrants(dataframe, colonne):
    q1 = dataframe[colonne].quantile(0.25)
    q3 = dataframe[colonne].quantile(0.75)
    iqr = q3 - q1
    borne_basse = q1 - 1.5 * iqr
    borne_haute = q3 + 1.5 * iqr
    n = dataframe[(dataframe[colonne] < borne_basse) | (dataframe[colonne] > borne_haute)].shape[0]
    print(f"{colonne:<18} | borne basse: {borne_basse:>8.2f} | borne haute: {borne_haute:>8.2f} | aberrants: {n}")

for col in ['Age', 'Purchase_Amount', 'Rating']:
    if col in df.columns:
        detecter_aberrants(df, col)

print("\nVérification des valeurs logiques :")
if 'Age' in df.columns:
    print(f"Âges invalides (<18 ou >100) : {df[(df['Age'] < 18) | (df['Age'] > 100)].shape[0]}")
if 'Purchase_Amount' in df.columns:
    print(f"Montants invalides (<=0)     : {df[df['Purchase_Amount'] <= 0].shape[0]}")
if 'Rating' in df.columns:
    print(f"Notes invalides (hors 1-5)   : {df[(df['Rating'] < 1) | (df['Rating'] > 5)].shape[0]}")
if 'Repeat_Customer' in df.columns:
    print(f"Repeat_Customer invalide     : {df[~df['Repeat_Customer'].isin(['Yes', 'No'])].shape[0]}")

print("\n" + "=" * 60)
print("RÉSUMÉ FINAL DU NETTOYAGE")
print("=" * 60)
print(f"Lignes au départ       : {avant}")
print(f"Lignes après nettoyage : {len(df)}")
print(f"Colonnes               : {df.shape[1]}")
print("\nTypes de données :")
print(df.dtypes.to_string())

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Détection des valeurs aberrantes", fontsize=13, fontweight='bold')

for ax, col in zip(axes, ['Age', 'Purchase_Amount', 'Rating']):
    ax.boxplot(
        df[col].dropna(),
        patch_artist=True,
        boxprops=dict(facecolor='lightblue', color='black'),
        medianprops=dict(color='red', linewidth=2)
    )
    ax.set_title(f"Boxplot — {col}")
    ax.set_ylabel(col)

plt.tight_layout()
plt.savefig("nettoyage_aberrants.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nNettoyage terminé.")

# ══════════════════════════════════════════════════════════════
# PARTIE 1 — MEILLEURS CLIENTS
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PARTIE 1 — MEILLEURS CLIENTS")
print("=" * 60)

fideles = df[df['Repeat_Customer'] == 'Yes'].copy()
non_fideles = df[df['Repeat_Customer'] == 'No'].copy()

print("\nTop 10 clients par montant d'achat :")
cols_top = [c for c in ['Customer_ID', 'Age', 'Purchase_Amount', 'Rating', 'Repeat_Customer'] if c in df.columns]
print(df.nlargest(10, 'Purchase_Amount')[cols_top].to_string(index=False))

print(f"\nClients fidèles        : {len(fideles)} ({len(fideles) / len(df) * 100:.1f}%)")
print(f"Dépense moyenne fidèles     : ${fideles['Purchase_Amount'].mean():.2f}")
print(f"Dépense moyenne non-fidèles : ${non_fideles['Purchase_Amount'].mean():.2f}")

age_spend = df.groupby('Age_Group', observed=True)['Purchase_Amount'].mean().sort_values(ascending=False)
print("\nDépense moyenne par groupe d'âge :")
print(age_spend.round(2).to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("PARTIE 1 — Meilleurs Clients", fontsize=13, fontweight='bold')

age_spend.plot(kind='bar', ax=axes[0], edgecolor='black')
axes[0].set_title("Dépense moyenne par groupe d'âge")
axes[0].set_xlabel("Groupe d'âge")
axes[0].set_ylabel("Montant moyen ($)")
axes[0].tick_params(axis='x', rotation=0)

df.groupby('Repeat_Customer', observed=True)['Purchase_Amount'].mean().plot(
    kind='bar', ax=axes[1], edgecolor='black'
)
axes[1].set_title("Fidèles vs Non-fidèles")
axes[1].set_xlabel("Client fidèle ?")
axes[1].set_ylabel("Montant moyen ($)")
axes[1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig("partie1_meilleurs_clients.png", dpi=150, bbox_inches="tight")
plt.show()

# ══════════════════════════════════════════════════════════════
# PARTIE 2 — POURQUOI CERTAINS PARTENT ?
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PARTIE 2 — POURQUOI CERTAINS PARTENT ?")
print("=" * 60)

print(f"\nNote moyenne fidèles     : {fideles['Rating'].mean():.2f}")
print(f"Note moyenne non-fidèles : {non_fideles['Rating'].mean():.2f}")

rating_cross = pd.crosstab(df['Rating'], df['Repeat_Customer'], normalize='columns') * 100
print("\nRépartition des notes (%) :")
print(rating_cross.round(1).to_string())

churn_age = df.groupby('Age_Group', observed=True)['Repeat_Customer'].apply(
    lambda x: (x == 'Yes').sum() / len(x) * 100
).round(1)

print("\nTaux de fidélité par groupe d'âge (%) :")
print(churn_age.to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("PARTIE 2 — Pourquoi Certains Partent ?", fontsize=13, fontweight='bold')

rating_cross.plot(kind='bar', ax=axes[0], edgecolor='black')
axes[0].set_title("Notes : Fidèles vs Non-fidèles")
axes[0].set_xlabel("Note")
axes[0].set_ylabel("% de clients")
axes[0].tick_params(axis='x', rotation=0)
axes[0].legend(title="Client fidèle")

churn_age.plot(kind='bar', ax=axes[1], edgecolor='black')
axes[1].set_title("Taux de fidélité par âge")
axes[1].set_xlabel("Groupe d'âge")
axes[1].set_ylabel("% clients fidèles")
axes[1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig("partie2_pourquoi_ils_partent.png", dpi=150, bbox_inches="tight")
plt.show()

# ══════════════════════════════════════════════════════════════
# PARTIE 3 — MEILLEURS PRODUITS
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PARTIE 3 — MEILLEURS PRODUITS")
print("=" * 60)

ca_cat = df.groupby('Category', observed=True)['Purchase_Amount'].sum().sort_values(ascending=False)
note_cat = df.groupby('Category', observed=True)['Rating'].mean().sort_values(ascending=False)
fidel_cat = df.groupby('Category', observed=True)['Repeat_Customer'].apply(
    lambda x: (x == 'Yes').sum() / len(x) * 100
).sort_values(ascending=False)

print("\nChiffre de ventes par catégorie :")
print(ca_cat.round(2).to_string())

print("\nNote moyenne par catégorie :")
print(note_cat.round(2).to_string())

print("\nTaux de fidélité par catégorie :")
print(fidel_cat.round(1).to_string())

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("PARTIE 3 — Meilleurs Produits", fontsize=13, fontweight='bold')

ca_cat.plot(kind='bar', ax=axes[0], edgecolor='black')
axes[0].set_title("Ventes par catégorie")
axes[0].set_ylabel("Ventes totales ($)")
axes[0].tick_params(axis='x', rotation=45)

note_cat.plot(kind='bar', ax=axes[1], edgecolor='black')
axes[1].set_title("Note moyenne par catégorie")
axes[1].set_ylabel("Note moyenne")
axes[1].tick_params(axis='x', rotation=45)

fidel_cat.plot(kind='bar', ax=axes[2], edgecolor='black')
axes[2].set_title("Fidélité par catégorie (%)")
axes[2].set_ylabel("% clients fidèles")
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("partie3_meilleurs_produits.png", dpi=150, bbox_inches="tight")
plt.show()

# ══════════════════════════════════════════════════════════════
# MACHINE LEARNING — PRÉDICTION FIDÉLITÉ
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MACHINE LEARNING — PRÉDICTION FIDÉLITÉ")
print("=" * 60)

df_ml = df.copy()

# Encodage de la cible
target_encoder = LabelEncoder()
df_ml['Repeat_Customer'] = target_encoder.fit_transform(df_ml['Repeat_Customer'])

# Gestion intelligente de Purchase_Date
if 'Purchase_Date' in df_ml.columns:
    df_ml['Purchase_Date'] = pd.to_datetime(df_ml['Purchase_Date'], errors='coerce')
    df_ml['Purchase_Month'] = df_ml['Purchase_Date'].dt.month
    df_ml['Purchase_Day'] = df_ml['Purchase_Date'].dt.day
    df_ml['Purchase_DayOfWeek'] = df_ml['Purchase_Date'].dt.dayofweek
    df_ml = df_ml.drop(columns=['Purchase_Date'])

# Suppression colonnes à forte cardinalité / inutiles
cols_to_drop = ['Customer_ID', 'City']
df_ml = df_ml.drop(columns=[c for c in cols_to_drop if c in df_ml.columns], errors='ignore')

# Encodage catégoriel
categorical_cols_ml = df_ml.select_dtypes(include=['object', 'category']).columns.tolist()
categorical_cols_ml = [c for c in categorical_cols_ml if c != 'Repeat_Customer']

df_ml = pd.get_dummies(df_ml, columns=categorical_cols_ml, drop_first=True)

X = df_ml.drop(columns=['Repeat_Customer'])
y = df_ml['Repeat_Customer']

print(f"\nNombre de lignes     : {X.shape[0]}")
print(f"Nombre de variables  : {X.shape[1]}")
print("\nRépartition de la cible :")
print(y.value_counts().to_string())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

modeles = {
    'Régression Logistique': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

resultats = {}

for nom, modele in modeles.items():
    if nom == 'Régression Logistique':
        Xtr, Xte = X_train_sc, X_test_sc
    else:
        Xtr, Xte = X_train, X_test

    modele.fit(Xtr, y_train)
    y_pred = modele.predict(Xte)
    y_proba = modele.predict_proba(Xte)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    resultats[nom] = {
        'modele': modele,
        'acc': acc,
        'auc': auc,
        'y_pred': y_pred,
        'y_proba': y_proba
    }

    print(f"\n{nom}")
    print(f"Accuracy : {acc * 100:.2f}%")
    print(f"AUC      : {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Non-fidèle', 'Fidèle']))

meilleur = max(resultats, key=lambda x: resultats[x]['auc'])
print(f"\nMeilleur modèle : {meilleur}")
print(f"AUC             : {resultats[meilleur]['auc']:.4f}")
print(f"Accuracy        : {resultats[meilleur]['acc'] * 100:.2f}%")

rf_model = resultats['Random Forest']['modele']
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

print("\nTop 15 variables importantes (Random Forest) :")
print(importances.head(15).round(4).to_string())

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Machine Learning — Prédiction Fidélité", fontsize=13, fontweight='bold')

noms = list(resultats.keys())
x = np.arange(len(noms))

axes[0].bar(x - 0.2, [resultats[n]['acc'] * 100 for n in noms], width=0.4, label='Accuracy', edgecolor='black')
axes[0].bar(x + 0.2, [resultats[n]['auc'] * 100 for n in noms], width=0.4, label='AUC-ROC', edgecolor='black')
axes[0].set_title("Comparaison des modèles")
axes[0].set_ylabel("Score (%)")
axes[0].set_xticks(x)
axes[0].set_xticklabels(noms, rotation=15, ha='right')
axes[0].legend()
axes[0].set_ylim(0, 110)

cm = confusion_matrix(y_test, resultats[meilleur]['y_pred'])
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    ax=axes[1],
    xticklabels=['Non-fidèle', 'Fidèle'],
    yticklabels=['Non-fidèle', 'Fidèle']
)
axes[1].set_title(f"Matrice de confusion\n({meilleur})")
axes[1].set_xlabel("Prédit")
axes[1].set_ylabel("Réel")

for nom, res in resultats.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
    axes[2].plot(fpr, tpr, label=f"{nom} (AUC={res['auc']:.2f})")

axes[2].plot([0, 1], [0, 1], 'k--', label='Aléatoire')
axes[2].set_title("Courbes ROC")
axes[2].set_xlabel("Taux de faux positifs")
axes[2].set_ylabel("Taux de vrais positifs")
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig("ml_prediction_fidelite.png", dpi=150, bbox_inches="tight")
plt.show()

# ══════════════════════════════════════════════════════════════
# SEGMENTATION CLIENT — KMEANS
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SEGMENTATION CLIENT — KMEANS")
print("=" * 60)

seg = df.copy()

# Feature engineering
seg['High_Spender'] = (seg['Purchase_Amount'] > seg['Purchase_Amount'].mean()).astype(int)
seg['Low_Rating'] = (seg['Rating'] <= 2).astype(int)
seg['Young'] = (seg['Age'] < 30).astype(int)

# Encodage fidélité
le_seg = LabelEncoder()
seg['Repeat_Customer_Enc'] = le_seg.fit_transform(seg['Repeat_Customer'])

# Gestion correcte de Purchase_Date
if 'Purchase_Date' in seg.columns:
    seg['Purchase_Date'] = pd.to_datetime(seg['Purchase_Date'], errors='coerce')
    seg['Purchase_Month'] = seg['Purchase_Date'].dt.month
    seg['Purchase_Day'] = seg['Purchase_Date'].dt.day
    seg['Purchase_DayOfWeek'] = seg['Purchase_Date'].dt.dayofweek
    seg = seg.drop(columns=['Purchase_Date'])

# Suppression de City
if 'City' in seg.columns:
    seg = seg.drop(columns=['City'])

# Colonnes à encoder
cols_encode = [c for c in ['Category', 'Gender', 'Payment_Method', 'Discount_Applied', 'Age_Group'] if c in seg.columns]
seg = pd.get_dummies(seg, columns=cols_encode, drop_first=True)

# Features finales
features_seg = [
    'Age',
    'Purchase_Amount',
    'Rating',
    'Repeat_Customer_Enc',
    'High_Spender',
    'Low_Rating',
    'Young'
]

for col in ['Purchase_Month', 'Purchase_Day', 'Purchase_DayOfWeek']:
    if col in seg.columns:
        features_seg.append(col)

dummy_prefixes = ('Category_', 'Gender_', 'Payment_Method_', 'Discount_Applied_', 'Age_Group_')
features_seg += [col for col in seg.columns if col.startswith(dummy_prefixes)]

X_seg = seg[features_seg].copy()

print("\nVariables utilisées pour la segmentation :")
print(features_seg)
print(f"\nNombre de variables de segmentation : {len(features_seg)}")

# Standardisation
scaler_seg = StandardScaler()
X_scaled = scaler_seg.fit_transform(X_seg)

# Méthode du coude
inertia = []
k_values = range(2, 9)

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(list(k_values), inertia, marker='o')
plt.title("Méthode du coude")
plt.xlabel("Nombre de clusters")
plt.ylabel("Inertie")
plt.grid(True)
plt.savefig("segmentation_elbow.png", dpi=150, bbox_inches="tight")
plt.show()

# Silhouette score
print("\nSilhouette score par nombre de clusters :")
best_k = None
best_score = -1

for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_k = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels_k)
    print(f"k = {k} -> silhouette = {score:.4f}")
    if score > best_score:
        best_score = score
        best_k = k

print(f"\nMeilleur k selon silhouette score : {best_k}")

# Clustering final
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
seg['Cluster'] = kmeans_final.fit_predict(X_scaled)

# Résumé des clusters
resume_clusters = seg.groupby('Cluster')[[
    'Age',
    'Purchase_Amount',
    'Rating',
    'Repeat_Customer_Enc',
    'High_Spender',
    'Low_Rating',
    'Young'
]].mean().round(2)

print("\nRésumé des clusters :")
print(resume_clusters.to_string())

print("\nTaille des clusters :")
print(seg['Cluster'].value_counts().sort_index().to_string())

# Création des segments business
global_purchase_mean = seg['Purchase_Amount'].mean()
global_fidelity_mean = seg['Repeat_Customer_Enc'].mean()
global_young_mean = seg['Young'].mean()
global_low_rating_mean = seg['Low_Rating'].mean()

mapping = {}

for cl in resume_clusters.index:
    montant = resume_clusters.loc[cl, 'Purchase_Amount']
    fidelite = resume_clusters.loc[cl, 'Repeat_Customer_Enc']
    jeune = resume_clusters.loc[cl, 'Young']
    faible_note = resume_clusters.loc[cl, 'Low_Rating']

    if montant >= global_purchase_mean and fidelite >= global_fidelity_mean:
        mapping[cl] = 'Clients premium fidèles 💎'
    elif montant >= global_purchase_mean and jeune >= global_young_mean:
        mapping[cl] = 'Jeunes gros acheteurs 🚀'
    elif faible_note >= global_low_rating_mean:
        mapping[cl] = 'Clients insatisfaits ⚠️'
    elif fidelite >= global_fidelity_mean:
        mapping[cl] = 'Clients fidèles réguliers 👍'
    else:
        mapping[cl] = 'Clients standards'

# Création OBLIGATOIRE de Segment
seg['Segment'] = seg['Cluster'].map(mapping)

print("\nDistribution des segments :")
print(seg['Segment'].value_counts().to_string())

# PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

seg['PCA1'] = X_pca[:, 0]
seg['PCA2'] = X_pca[:, 1]

plt.figure(figsize=(9, 6))
plt.scatter(seg['PCA1'], seg['PCA2'], c=seg['Cluster'])
plt.title("Segmentation clients — Projection PCA")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)
plt.savefig("segmentation_pca.png", dpi=150, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 5))
for cl in sorted(seg['Cluster'].unique()):
    subset = seg[seg['Cluster'] == cl]
    plt.scatter(subset['Age'], subset['Purchase_Amount'], label=f"Cluster {cl}", alpha=0.6)

plt.title("Segmentation KMeans des clients")
plt.xlabel("Age")
plt.ylabel("Purchase_Amount")
plt.legend()
plt.grid(True)
plt.savefig("segmentation_kmeans_adaptee.png", dpi=150, bbox_inches="tight")
plt.show()

seg.to_csv("segmentation_clients_finale.csv", index=False)

# ══════════════════════════════════════════════════════════════
# EXEMPLE DE PRÉDICTION
# ══════════════════════════════════════════════════════════════
nouveau = pd.DataFrame([{
    'Age': 35,
    'Purchase_Amount': 300.0,
    'Rating': 4,
    'Purchase_Month': 1,
    'Purchase_Day': 15,
    'Purchase_DayOfWeek': 2
}])

for col in X.columns:
    if col not in nouveau.columns:
        nouveau[col] = 0

nouveau = nouveau[X.columns]

if meilleur == 'Régression Logistique':
    pred = resultats[meilleur]['modele'].predict(scaler.transform(nouveau))[0]
    proba = resultats[meilleur]['modele'].predict_proba(scaler.transform(nouveau))[0]
else:
    pred = resultats[meilleur]['modele'].predict(nouveau)[0]
    proba = resultats[meilleur]['modele'].predict_proba(nouveau)[0]

print("\nExemple de prédiction :")
print("Client test : 35 ans, achat = 300$, note = 4/5")
print(f"Prédiction  : {'Fidèle' if pred == 1 else 'Non-fidèle'}")
print(f"Probabilité : Fidèle = {proba[1] * 100:.1f}% | Non-fidèle = {proba[0] * 100:.1f}%")

# ══════════════════════════════════════════════════════════════
# RÉSUMÉ FINAL
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("RÉSUMÉ FINAL")
print("=" * 60)

print(f"\nNombre final de lignes : {len(df)}")
print(f"Colonnes disponibles   : {list(df.columns)}")

print("\nAperçu des données nettoyées :")
print(df.head().to_string())

print("\nTop segments finaux :")
if 'Segment' in seg.columns:
    print(seg['Segment'].value_counts().to_string())
else:
    print("La colonne 'Segment' n'existe pas encore.")

print("\nFichiers générés :")
print("- nettoyage_aberrants.png")
print("- partie1_meilleurs_clients.png")
print("- partie2_pourquoi_ils_partent.png")
print("- partie3_meilleurs_produits.png")
print("- ml_prediction_fidelite.png")
print("- segmentation_elbow.png")
print("- segmentation_pca.png")
print("- segmentation_kmeans_adaptee.png")
print("- segmentation_clients_finale.csv")

print("\nScript terminé avec succès.")

# ══════════════════════════════════════════════════════════════
# PCA — VISUALISATION PROPRE DES CLUSTERS
# ══════════════════════════════════════════════════════════════
from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

seg['PCA1'] = X_pca[:, 0]
seg['PCA2'] = X_pca[:, 1]

print("\nVariance expliquée par la PCA :")
print(f"PCA1 : {pca.explained_variance_ratio_[0]*100:.2f}%")
print(f"PCA2 : {pca.explained_variance_ratio_[1]*100:.2f}%")
print(f"Total : {(pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1])*100:.2f}%")

plt.figure(figsize=(10, 6))
for cl in sorted(seg['Cluster'].unique()):
    subset = seg[seg['Cluster'] == cl]
    plt.scatter(subset['PCA1'], subset['PCA2'], label=f"Cluster {cl}", alpha=0.6)

plt.title("Projection PCA des clusters clients")
plt.xlabel("Composante principale 1")
plt.ylabel("Composante principale 2")
plt.legend()
plt.grid(True)
plt.savefig("segmentation_pca_propre.png", dpi=150, bbox_inches="tight")
plt.show()