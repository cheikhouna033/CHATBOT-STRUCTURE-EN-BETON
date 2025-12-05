import streamlit as st
import os
import re

# ============================================
# ✔ Corrections texte inversé
# ============================================
def is_reversed(text):
    words = text.split()
    reversed_count = sum(1 for w in words if w[::-1].lower() in text.lower())
    return reversed_count > len(words) * 0.5

def fix_reversed_text(text):
    return text[::-1]


# ============================================
# ✔ Extraction PDF → TXT
# ============================================
def extract_pdf_to_txt(pdf_path, txt_path):

    if os.path.exists(txt_path):
        return

    full_text = ""
    pdfplumber_available = False
    pypdf2_available = False

    # Tentative import pdfplumber
    try:
        import pdfplumber
        pdfplumber_available = True
    except:
        pass

    # Tentative import PyPDF2
    try:
        from PyPDF2 import PdfReader
        pypdf2_available = True
    except:
        pass

    if not pdfplumber_available and not pypdf2_available:
        st.error("Aucun parseur PDF disponible. Installez pdfplumber ou PyPDF2.")
        return

    # --- Extraction principale ---
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

    # Écriture du fichier texte
    with open(txt_path, "w", encoding="utf-8", errors="ignore") as f:
        f.write(full_text)


# ============================================
# ✔ Import NLTK + prétraitement
# ============================================
import nltk

for pkg in ["punkt", "stopwords", "wordnet"]:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ============================================
# ✔ Prétraitement (style GOMYCODE amélioré)
# ============================================
def preprocess(sentence):
    words = word_tokenize(sentence)

    sw = set(stopwords.words("french"))
    punct = set(".,;:!?()[]{}'\"-–")

    words = [w.lower() for w in words if w.lower() not in sw and w not in punct]

    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]

    return words


# ============================================
# ✔ Similarité Jaccard (GOMYCODE)
# ============================================
def jaccard_similarity(a, b):
    a, b = set(a), set(b)
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


# ============================================
# ✔ Chatbot
# ============================================
def chatbot(question, sentences, corpus):
    return find_best_sentence(question, sentences, corpus)


# ============================================
# ✔ Application Streamlit
# ============================================
def main():
    st.title("🤖 Chatbot – Formation ARCHE (Structures)")

    pdf_path = "Formation_Arche.pdf"
    txt_path = "formation_arche.txt"

    # Extraction PDF si pas déjà fait
    extract_pdf_to_txt(pdf_path, txt_path)

    # Lecture fichier texte
    if not os.path.exists(txt_path):
        st.error("Impossible de charger le fichier texte extrait du PDF.")
        return

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        raw_text = f.read()

    sentences = sent_tokenize(raw_text)
    corpus = [preprocess(s) for s in sentences]

    question = st.text_input("Posez votre question sur ARCHE :")

    if st.button("🔎 Rechercher"):
        if question.strip() == "":
            st.warning("Veuillez entrer une question.")
        else:
            response = chatbot(question, sentences, corpus)
            st.markdown("### 📘 Réponse")
            st.write(response)


if __name__ == "__main__":
    main()
