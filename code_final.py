
# Importer les packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import joblib
import nltk
nltk.download('stopwords') # Télécharger le package stopwords
nltk.download('wordnet')
from nltk.corpus import stopwords # Importer le package stopwords



# Afficher les données
data=pd.read_csv('C:/Users/bmd tech/projet_master/reddit_bad.csv')
df= pd.read_csv('C:/Users/bmd tech/projet_master/reddit_good.csv')



# Dimension des données
df.shape


print(data['selftext'].apply(type).value_counts())
print(df['title'].apply(type).value_counts())



# Afficher la première ligne de la colonne text
df['title'][0]

# Afficher la deuxième ligne de la colonne text
data['selftext'][0]


# Fonction supprimant les links
def remove_links(title):
    title= re.sub(r'http:?//\S+ | https:?//\S+','',title)
    return title

# Fonction supprimant les @username
def remove_users(title):
    title = re.sub(r'@[\w\-._]+','',title)
    return title

# Supprimer les adresses emails
def email_address(text):
  title= re.sub(r'@[\w\-._]+','',text)
  return title


import contractions
# Fonction étendant les contractions
def contraction(title):
    expanded_all = []
    for word in title.split():
        expanded_all.append(contractions.fix(word)) # utiliser la fonction fix de contractions

    expand = ' '.join(expanded_all)
    return expand

# Supprimer html caractères
def clean_html(text):
   title = re.sub(r'&\w+','',text)
   return title

# Remplacer tout ce qui n'est chaines de caractères alphabétiques et espace par ' '
def alpha_b(text):
   title = re.sub(r'^a-zA-Z\s]+',' ',text)
   return title

def crochet(text):
   title= re.sub(r'\[.*?\]', '', text)
   return title

# Fonction remplaçant les espaces multiples et convertissant majuscules en minuscules
def lower(text):
    title = re.sub(r'\s(2,)',' ' ,text)
    return title

# Supprimer les espaces en début et fin de tweet
def clean_space(text):
    title = re.sub(r'^\s|\s$',' ' ,text)
    return title

# Fonction supprimant les stopwords
def remove_stopwords(text):
    Stopwords = stopwords.words('english')
    title= ' '.join([word for word in text.split() if word not in Stopwords])
    return title

# Lemmatization
from nltk.stem import WordNetLemmatizer
lemma=WordNetLemmatizer()
def lem_sw(text):
    title = [lemma.lemmatize(word) for word in text.split()]
    title = " ".join(title)
    return title



# Appliquer la fonction remove_users
data['new_title'] = data.title.apply(func =remove_users)
# Appliquer la fonction remove_links
data['new_title'] = data.new_title.apply(func =remove_links)
# Appliquer la fonction email_address
data['new_title'] = data.new_title.apply(func =email_address)
# Appliquer la fonction remove_contraction
data['new_title'] = data.new_title.apply(func = contraction)
# Appliquer la fonction clean_html
data['new_title'] = data.new_title.apply(func =clean_html)
# Appliquer la fonction alpha_b
data['new_title'] = data.new_title.apply(func =alpha_b)
# Appliquer la fonction crochet
data['new_title'] = data.new_title.apply(func =crochet)
# Appliquer la fonction lower
data['new_title'] = data.new_title.apply(func =lower)
# Appliquer la fonction clean_space
data['new_title'] = data.new_title.apply(func =clean_space)
# Appliquer la fonction remove_stopwords
data['new_title'] = data.new_title.apply(func =remove_stopwords)
# Appliquer la fonction lem_sw
data['new_title'] = data.new_title.apply(func =lem_sw)

# Appliquer la fonction remove_users
df['new_title'] = df.title.apply(func =remove_users)
# Appliquer la fonction remove_links
df['new_title'] = df.new_title.apply(func =remove_links)
# Appliquer la fonction email_address
df['new_title'] = df.new_title.apply(func =email_address)
# Appliquer la fonction remove_contraction
df['new_title'] = df.new_title.apply(func = contraction)
# Appliquer la fonction clean_html
df['new_title'] = df.new_title.apply(func =clean_html)
# Appliquer la fonction alpha_b
df['new_title'] = df.new_title.apply(func =alpha_b)
# Appliquer la fonction crochet
df['new_title'] = df.new_title.apply(func =crochet)
# Appliquer la fonction lower
df['new_title'] = df.new_title.apply(func =lower)
# Appliquer la fonction clean_space
df['new_title'] = df.new_title.apply(func =clean_space)
# Appliquer la fonction remove_stopwords
df['new_title'] = df.new_title.apply(func =remove_stopwords)
# Appliquer la fonction lem_sw
df['new_title'] = df.new_title.apply(func =lem_sw)

# Afficher la ligne d'index 0
df['new_title'][1200]

df.isnull().sum()

df

columns_to_drop = ['title', 'selftext', 'created_utc', 'author', 'subreddit']
# Garder uniquement les colonnes existantes dans le DataFrame
columns_to_drop = [col for col in columns_to_drop if col in data.columns]

data = data.drop(columns=columns_to_drop)
data

columns_to_drop = ['title', 'selftext', 'created_utc', 'author', 'subreddit']
# Garder uniquement les colonnes existantes dans le DataFrame
columns_to_drop = [col for col in columns_to_drop if col in df.columns]

df = df.drop(columns=columns_to_drop)
df

data["label"] = 1  # 1 pour violent
df["label"] = 0  # 0 pour non violent
combined_df = pd.concat([data, df], ignore_index=True)

combined_df

combined_df["label"].value_counts()

from sklearn.utils import resample

# Texte violent (classe minoritaire)
violent = combined_df[combined_df["label"] == 1]

# Texte non violent (classe majoritaire)
non_violent = combined_df[combined_df["label"] == 0]

# Sur-échantillonnage pour équilibrer
non_violent_upsampled = resample(non_violent,
                              replace=True,  # Permet de dupliquer les données existantes
                              n_samples=len(violent),  # Nombre d'échantillons à égaler à la classe majoritaire
                              random_state=42)

# Combiner les deux classes
balanced_df = pd.concat([ violent, non_violent_upsampled])

# Vérifier la répartition des classes
print(balanced_df["label"].value_counts())

balanced_df=combined_df.dropna()

balanced_df.shape

# Vectorisation"""


# Importer train_test_split
from sklearn.model_selection import train_test_split

X = balanced_df["new_title"]  # Caractéristiques (textes)
y = balanced_df["label"]  # Labels (0 ou 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Taille des données d'entraînement : {len(X_train)}")
print(f"Taille des données de test : {len(X_test)}")

from sklearn.feature_extraction.text import TfidfVectorizer

# Initialiser le vectoriseur TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)  # Ajuste max_features si besoin

# Transformer les données
X_train_vec = vectorizer.fit_transform(X_train).toarray()
X_test_vec = vectorizer.transform(X_test).toarray()

print(f"Taille des vecteurs TF-IDF (entraînement) : {X_train_vec.shape}")
print(f"Taille des vecteurs TF-IDF (test) : {X_test_vec.shape}")

from sklearn.feature_selection import chi2
import numpy as np

# Appliquer le test du Chi-2
chi2_scores, p_values = chi2(X_train_vec, y_train)

# Associer les scores aux termes
features = vectorizer.get_feature_names_out()
chi2_results = pd.DataFrame({
    "Feature": features,
    "Chi2_Score": chi2_scores,
    "P_Value": p_values
})

# Trier par pertinence
chi2_results = chi2_results.sort_values(by="Chi2_Score", ascending=False)

# Afficher les termes les plus pertinents
print("Les termes les plus pertinents selon le Chi-2 :")
print(chi2_results.head(10))

# Filtrer les termes pertinents
relevant_features = chi2_results[chi2_results["P_Value"] < 0.05]

print(f"Nombre de termes pertinents sélectionnés : {len(relevant_features)}")
print(relevant_features.head())

# Créer un nouveau vecteur TF-IDF avec les termes pertinents
selected_features = relevant_features["Feature"].tolist()

# Reconstruire le vectoriseur avec les termes sélectionnés
vectorizer_selected = TfidfVectorizer(vocabulary=selected_features)

# Transformer les données d'entraînement et de test avec les termes sélectionnés
X_train_selected = vectorizer_selected.fit_transform(X_train)
X_test_selected = vectorizer_selected.transform(X_test)

print(f"Taille de la nouvelle matrice d'entraînement : {X_train_selected.shape}")
print(f"Taille de la nouvelle matrice de test : {X_test_selected.shape}")



"""RANDOM FOREST"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Initialiser le modèle Random Forest
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Entraîner le modèle
model_rf.fit(X_train_selected, y_train)

print("Random Forest entraîné avec succès !")

# Faire des prédictions
y_pred = model_rf.predict(X_test_selected)

# Afficher les métriques
print("Rapport de classification :")
print(classification_report(y_test, y_pred))

# Précision globale
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision du modèle : {accuracy:.2f}")

# Exemple pour un SVM
joblib.dump(model_rf, "rf_model.pkl")

# Exemple de textes à prédire
new_texts = [
    "I see diomestic violence everday.",
    "Is it good to be beaten ",# Exemple violent
    "She enjoys spending time with friends",
    "Feel like she is emotionally abuse.",
    "Life is great those days",
    "I am happy",
    "I am in a safe relationship",
    "My dad used to beat me after school"
]

# Transformer les nouveaux textes
new_texts_vec =  vectorizer_selected.transform(new_texts)

# Faire des prédictions
predictions = model_rf.predict(new_texts_vec)

# Afficher les résultats
for text, pred in zip(new_texts, predictions):
    label = "Potiental danger" if pred == 1 else "No danger"
    print(f"Texte : '{text}' -> Prédiction : {label}")


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Prédire sur les données de test
y_pred = model_rf.predict(X_test_selected)  # Si tu utilises un modèle LSTM ou un autre modèle
if hasattr(y_pred[0], "__len__"):  # Si le modèle donne des probabilités
    y_pred = (y_pred > 0.5).astype(int)  # Convertir les probabilités en classes binaires

# Calculer la matrice de confusion
cm = confusion_matrix(y_test, y_pred)

# Afficher la matrice de confusion
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No danger", "Potential danger"])
disp.plot(cmap=plt.cm.BuGn)
plt.title("Matrice de confusion")
plt.show()



"""#DECISION TREE"""

from sklearn.tree import DecisionTreeClassifier
# Initialiser le modèle Decision Tree
model_dt = DecisionTreeClassifier(random_state=42)

# Entraîner le modèle
model_dt.fit(X_train_selected, y_train)

print("Modèle Decision Tree entraîné avec succès !")

# Faire des prédictions
y_pred = model_dt.predict(X_test_selected)

# Afficher les métriques
print("Rapport de classification :")
print(classification_report(y_test, y_pred))
# Précision globale
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision du modèle : {accuracy:.2f}")

# Exemple de textes à prédire
new_texts = [
    "I see diomestic violence everday.",
    "Is it good to be beaten ",# Exemple violent
    "She enjoys spending time with friends",
    "Feel like she is emotionally abuse.",
    "Life is great those days",
    "I am happy",
    "I am in a safe relationship",
    "My dad used to beat me after school",# Exemple non violent
]

# Transformer les nouveaux textes
new_texts_vec = vectorizer_selected.transform(new_texts)

# Faire des prédictions
predictions = model_dt.predict(new_texts_vec)

# Afficher les résultats
for text, pred in zip(new_texts, predictions):
    label = "Potential danger" if pred == 1 else "No danger"
    print(f"Texte : '{text}' -> Prédiction : {label}")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Prédire sur les données de test
y_pred = model_dt.predict(X_test_selected)  # Si tu utilises un modèle LSTM ou un autre modèle
if hasattr(y_pred[0], "__len__"):  # Si le modèle donne des probabilités
    y_pred = (y_pred > 0.5).astype(int)  # Convertir les probabilités en classes binaires

# Calculer la matrice de confusion
cm = confusion_matrix(y_test, y_pred)

# Afficher la matrice de confusion
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No danger", "Potential danger"])
disp.plot(cmap=plt.cm.gnuplot)
plt.title("Matrice de confusion")
plt.show()

