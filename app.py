import streamlit as st
import os
import re

# ==========================================================
# 1) TELECHARGEMENT ROBUSTE DES RESSOURCES NLTK
# ==========================================================
import nltk

def ensure_nltk_resources():
    packages = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet"
    }
    for pkg, path in packages.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)

ensure_nltk_resources()

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ==========================================================
# 2) FONCTIONS : DETECTION ET CORRECTION TEXTE INVERSE
# ==========================================================
def is_reversed(text):
    words = text.split()
    reversed_count = sum(1 for w in words if w[::-1].lower() in text.lower())
    return reversed_count > len(words) * 0.5

def fix_reversed_text(text):
    return text[::-1]


# ==========================================================
# 3) EXTRACTION PDF → TXT
# ==========================================================
def extract_pdf_to_txt(pdf_path, txt_path):

    if os.path.exists(txt_path):
        return

    full_text = ""
    pdfplumber_available = False
    pypdf2_available = False

    # Tentative d'import
    try:
        import pdfplumber
        pdfplumber_available = True
    except:
        pass

    try:
        from PyPDF2 import PdfReader
        pypdf2_available = True
    except:
        pass

    if not pdfplumber_available and not pypdf2_available:
        st.error("Aucun parseur PDF installé. Installez pdfplumber ou PyPDF2.")
        return

    if pdfplumber_available:
        import pdfplumber
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
            except:
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

    # Sauvegarde
    with open(txt_path, "w", encoding="utf-8", errors="ignore") as f:
        f.write(full_text)


# ==========================================================
# 4) PRETRAITEMENT PAR PHRASE (style GOMYCODE amélioré)
# ==========================================================
def preprocess(sentence):
    words = word_tokenize(sentence)

    sw = set(stopwords.words("french"))
    punct = set(".,;:!?()[]{}'\"-–")

    words = [
        w.lower()
        for w in words
        if w.lower() not in sw and w not in punct
    ]

    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]

    return words


# ==========================================================
# 5) SIMILARITE JACCARD
# ==========================================================
def jaccard_similarity(a, b):
    a = set(a)
    b = set(b)
    if not a and not b:
        return 0
    return len(a.intersection(b)) / len(a.union(b))


def find_best_sentence(query, sentences, corpus):
    query_tokens = preprocess(query)

    best_sim = 0
    best_sentence = "Je n'ai trouvé aucune réponse pertinente."

    for sent, tokens in zip(sentences, corpus):
        sim = jaccard_similarity(query_tokens, tokens)
        if sim > best_sim:
            best_sim = sim
            best_sentence = sent

    return best_sentence


# ==========================================================
# 6) CHATBOT
# ==========================================================
def chatbot(question, sentences, corpus):
    return find_best_sentence(question, sentences, corpus)


# ==========================================================
# 7) INTERFACE STREAMLIT
# ==========================================================
def main():
    st.title("🤖 Chatbot – Formation ARCHE (Structures Béton Armé)")
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

    pdf_path = "Formation_Arche.pdf"
    txt_path = "formation_arche.txt"

    extract_pdf_to_txt(pdf_path, txt_path)

    if not os.path.exists(txt_path):
        st.error("Le fichier texte n’a pas pu être généré.")
        return

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        raw_text = f.read()

    sentences = sent_tokenize(raw_text)
    corpus = [preprocess(s) for s in sentences]

    question = st.text_input("Posez votre question sur Arche Ossature :")

    if st.button("🔎 Rechercher"):
        if not question.strip():
            st.warning("Veuillez entrer une question.")
        else:
            response = chatbot(question, sentences, corpus)
            st.markdown("### 📘 Réponse :")
            st.write(response)


if __name__ == "__main__":
    main()
