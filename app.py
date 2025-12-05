import streamlit as st
import os
import re

# ----------------------------
# Vérification des dépendances
# ----------------------------

# PDF parsers
pdfplumber = None
pypdf2_available = False

try:
    import pdfplumber
except ModuleNotFoundError:
    st.warning("pdfplumber non installé. Tentative d'utiliser PyPDF2.")
    try:
        from PyPDF2 import PdfReader
        pypdf2_available = True
    except ModuleNotFoundError:
        st.error(
            "Aucun parseur PDF installé. Installez 'pdfplumber' ou 'PyPDF2' dans requirement.txt."
        )
        raise

# NLTK
try:
    import nltk
except ModuleNotFoundError:
    st.error(
        "Le package 'nltk' n'est pas installé. Ajoutez-le dans requirement.txt et redeployez."
    )
    raise

# Télécharger les corpus nécessaires si absent
for corpus in ["stopwords", "words"]:
    try:
        nltk.data.find(f"corpora/{corpus}")
    except LookupError:
        nltk.download(corpus, quiet=True)

from nltk.corpus import stopwords

# scikit-learn
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ModuleNotFoundError:
    st.error(
        "Le package 'scikit-learn' n'est pas installé. Ajoutez-le dans requirement.txt et redeployez."
    )
    raise

# ----------------------------
# Détection et correction du texte inversé
# ----------------------------
def is_reversed(text):
    words_list = text.split()
    reversed_count = sum(1 for w in words_list if w[::-1].lower() in text.lower())
    return reversed_count > len(words_list) * 0.5

def fix_reversed_text(text):
    return text[::-1]

# ----------------------------
# Extraction PDF → TXT
# ----------------------------
def extract_pdf_to_txt(pdf_path, txt_path):
    if os.path.exists(txt_path):
        return

    full_text = ""

    if pdfplumber is not None:
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

    elif pypdf2_available:
        from PyPDF2 import PdfReader
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            try:
                text = page.extract_text()
            except Exception:
                text = None
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

    else:
        st.error("Aucun parseur PDF disponible.")
        raise RuntimeError("No PDF parser available")

    with open(txt_path, "w", encoding="utf-8", errors="ignore") as f:
        f.write(full_text)

    print("Extraction PDF → TXT créée.")

# ----------------------------
# Prétraitement du texte
# ----------------------------
def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[\.\?\!])\s+', text) if len(s.strip()) > 3]

french_dict = set([
    "la","le","les","des","du","de","et","entre","dans","utilisation","langue","logiciel",
    "document","production","possible","anglais","allemand","note","calcul","tracé","graphiques",
    "poutre","poteau","dalle","arche","ossature","structure","béton","armé","dimensionnement",
    "charge","charges","modèle","modélisation"
])

def fix_word_spacing(text):
    tokens = text.split()
    fixed_tokens = []

    for token in tokens:
        if len(token) > 15:
            result = []
            current = ""
            for char in token:
                current += char
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
    text = fix_word_spacing(text)

    raw = split_sentences(text)

    stop_words = set(stopwords.words("french"))
    cleaned = []

    for sent in raw:
        sent = fix_word_spacing(sent)
        tokens = re.findall(r'\w+', sent)
        tokens = [w for w in tokens if w not in stop_words]
        cleaned.append(" ".join(tokens))

    return raw, cleaned

# ----------------------------
# Similarité TF-IDF
# ----------------------------
def best_sentence_index(query, cleaned):
    vect = TfidfVectorizer()
    tfidf = vect.fit_transform(cleaned + [query])
    sim = cosine_similarity(tfidf[-1], tfidf[:-1])
    return sim.argmax()

def chatbot(query, raw, cleaned):
    idx = best_sentence_index(query, cleaned)
    return raw[idx]

# ----------------------------
# Application Streamlit
# ----------------------------
def main():
    st.title("🤖 Chatbot – Formation ARCHE (Structures)")

    with st.expander("ℹ️ Instructions et Utilité du Chatbot", expanded=True):
        st.markdown("""
### 🎯 Objectif
Aider à comprendre et utiliser **le logiciel Arche Ossature** à partir du PDF Formation_Arche.pdf.

### 🧠 Fonctionnalités
- Recherche de phrases pertinentes
- Explications sur modélisation, éléments béton armé, ferraillage
- Dimensionnement et règles BAEL / Eurocode

### ❓ Exemples
- "Qu'est-ce qu'un portique ?"
- "Comment modéliser un plancher dans Arche ?"
- "C’est quoi une poutre continue ?"

⚠️ Le chatbot ne remplace pas le logiciel ni les calculs réels.
""")

    pdf_path = "Formation_Arche.pdf"
    txt_path = "formation_arche.txt"

    extract_pdf_to_txt(pdf_path, txt_path)

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    raw, cleaned = preprocess(text)

    question = st.text_input("Posez votre question sur ARCHE :")

    if question:
        answer = chatbot(question, raw, cleaned)
        st.markdown("### 📘 Réponse")
        st.write(answer)

if __name__ == "__main__":
    main()
