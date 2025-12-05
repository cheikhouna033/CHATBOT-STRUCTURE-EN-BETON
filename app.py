# (Top of file)

import streamlit as st
import sys
import os
import re
import string

# Guarded import for pdfplumber so we can show a clear message in the UI.
try:
    import pdfplumber
except ModuleNotFoundError:
    st.error(
        "Missing Python package 'pdfplumber'.\n\n"
        "Fix: install the app dependencies and restart the app:\n\n"
        "  pip install pdfplumber\n\n"
        "Or add 'pdfplumber' to requirements.txt and redeploy (Streamlit Cloud)."
    )
    raise

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Try to import nltk and ensure required corpora are available.
try:
    import nltk
    # Make sure the corpora we use are available; download quietly if not found.
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('corpora/words')
    except LookupError:
        nltk.download('words', quiet=True)

    from nltk.corpus import stopwords, words
except ModuleNotFoundError:
    st.error(
        "Missing Python package 'nltk'.\n\n"
        "Fix: install the app dependencies and restart the app:\n\n"
        "  pip install -r requirements.txt\n\n"
        "If you're deploying on Streamlit Cloud, add requirements.txt to the repo root and redeploy."
    )
    raise


# ------------------------------------------------------------
# Détection et correction du texte inversé
# ------------------------------------------------------------
def is_reversed(text):
    words = text.split()
    reversed_count = sum(1 for w in words if w[::-1].lower() in text.lower())
    return reversed_count > len(words) * 0.5

def fix_reversed_text(text):
    return text[::-1]


# ------------------------------------------------------------
# Extraction PDF → TXT
# ------------------------------------------------------------
def extract_pdf_to_txt(pdf_path, txt_path):

    if os.path.exists(txt_path):
        return

    full_text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue

            text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
            lines = []

            for line in text.split("\n"):
                line = line.strip()
                if len(line) < 3:
                    continue

                if is_reversed(line):
                    line = fix_reversed_text(line)

                lines.append(line)

            full_text += "\n".join(lines) + "\n"

    with open(txt_path, "w", encoding="utf-8", errors="ignore") as f:
        f.write(full_text)

    print("Extraction PDF → formation_arche.txt créée.")


# ------------------------------------------------------------
# Prétraitement (sans punkt)
# ------------------------------------------------------------
def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[\.\?\!])\s+', text) if len(s.strip()) > 3]


french_dict = set([
    "la","le","les","des","du","de","et","entre","dans","utilisation","langue","logiciel",
    "document","production","possible","anglais","allemand","note","calcul","tracé","graphiques",
    "poutre","poteau","dalle","arche","ossature","structure","béton","armé","dimensionnement",
    "charge","charges","modèle","modélisation"
])
def fix_word_spacing(text):
    # Si le mot est trop long, on tente de le découper
    tokens = text.split()
    fixed_tokens = []

    for token in tokens:
        if len(token) > 15:  # mot trop long → probablement collé
            result = []
            current = ""

            for char in token:
                current += char
                # si le mot courant existe dans dictionnaire → on coupe
                if current.lower() in french_dict:
                    result.append(current)
                    current = ""
            if current:
                result.append(current)

            fixed_tokens.extend(result)
        else:
            fixed_tokens.append(token)

    return " ".join(fixed_tokens)

def preprocess(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)

    # Correction de mots collés AVANT segmentation
    text = fix_word_spacing(text)

    raw = split_sentences(text)

    stop_words = set(stopwords.words("french"))
    cleaned = []

    for sent in raw:
        # Correction de mots collés dans chaque phrase
        sent = fix_word_spacing(sent)

        tokens = re.findall(r'\w+', sent)
        tokens = [w for w in tokens if w not in stop_words]
        cleaned.append(" ".join(tokens))

    return raw, cleaned


# ------------------------------------------------------------
# Similarité TF-IDF
# ------------------------------------------------------------
def best_sentence_index(query, cleaned):
    vect = TfidfVectorizer()
    tfidf = vect.fit_transform(cleaned + [query])
    sim = cosine_similarity(tfidf[-1], tfidf[:-1])
    return sim.argmax()

def chatbot(query, raw, cleaned):
    idx = best_sentence_index(query, cleaned)
    return raw[idx]


# ------------------------------------------------------------
# APPLICATION STREAMLIT
# ------------------------------------------------------------
def main():

    st.title("🤖 Chatbot – Formation ARCHE (Structures)")

    # -------- PAGE D’ACCUEIL / INSTRUCTIONS --------
    with st.expander("ℹ️ **Instructions et Utilité du Chatbot**", expanded=True):
        st.markdown("""
### 🎯 **Objectif du chatbot**
Ce chatbot a été créé pour vous aider à comprendre et utiliser efficacement **le logiciel Arche Ossature** et son environnement pédagogique basé sur le document :

📘 *Formation_Arche.pdf* – Support de formation bâtiment et béton armé.

---

### 🧠 **Ce que fait le chatbot**
Il :
- recherche dans le PDF la phrase la plus pertinente
- vous fournit la définition, l'explication ou la procédure associée
- peut aider à comprendre des notions de :
  - modélisation sous ARCHE
  - éléments béton armé
  - dimensionnement et règles BAEL / Eurocode
  - principes des descentes de charges
  - notions de ferraillage
  - méthodologie de calcul structurel

---

### ❓ **Exemples de questions que vous pouvez poser**
- *"Qu'est-ce qu'un portique ?"*
- *"Comment modéliser un plancher dans Arche ?"*
- *"C’est quoi une poutre continue ?"*
- *"Comment fonctionne le ferraillage automatique ?"*
- *"Définition d'une charge linéique ?"*
- *"Comment exporter vers Arche Poutre ?"*

---

### 🛑 **Ce que le chatbot NE fait pas**
⚠️ Il ne :
- crée pas des plans
- ne fait pas de calcul automatique en temps réel
- ne remplace pas une vraie simulation ARCHE
- ne répond pas en dehors du contenu du PDF

Il se base **uniquement sur le texte de Formation_Arche.pdf**.

---

### 📝 **Comment formuler vos questions**
Pour de meilleurs résultats :
- écrivez des phrases courtes
- utilisez des termes techniques du bâtiment
- posez une question en lien avec le document

Exemples :
- *"Définition d'un poteau BA ?"*
- *"Rôle de la dalle dans un plancher ?"*

---

Bonne utilisation ! 😊
""")

    # -------- TRAITEMENT PDF --------
    pdf_path = "Formation_Arche.pdf"
    txt_path = "formation_arche.txt"

    extract_pdf_to_txt(pdf_path, txt_path)

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    raw, cleaned = preprocess(text)

    # -------- QUESTION UTILISATEUR --------
    question = st.text_input("Posez votre question sur ARCHE :")

    if question:
        answer = chatbot(question, raw, cleaned)
        st.markdown("### 📘 Réponse")
        st.write(answer)


if __name__ == "__main__":
    main()