import streamlit as st
import os
import re

# ==========================================================
# 1) IMPORTS SÉCURISÉS POUR ÉVITER LES CRASHS STREAMLIT CLOUD
# ==========================================================
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
except ModuleNotFoundError:
    st.error("❌ Le package NLTK n'est pas installé. Ajoutez-le dans requirements.txt et redeployez.")
    st.stop()


# Téléchargement robuste des ressources NLTK
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


# ==========================================================
# 2) DETECTION TEXTE INVERSE
# ==========================================================
def is_reversed(text):
    words = text.split()
    reversed_count = sum(1 for w in words if w[::-1].lower() in text.lower())
    return reversed_count > len(words) * 0.5


def fix_reversed_text(text):
    return text[::-1]


# ==========================================================
# 3) EXTRACTION PDF → TXT AVEC GESTION D’ERREURS
# ==========================================================
def extract_pdf_to_txt(pdf_path, txt_path):

    if os.path.exists(txt_path):
        return

    full_text = ""
    pdfplumber_available = False
    pypdf2_available = False

    try:
        import pdfplumber
        pdfplumber_available = True
    except:
        st.warning("pdfplumber indisponible, tentative avec PyPDF2…")

    try:
        from PyPDF2 import PdfReader
        pypdf2_available = True
    except:
        st.warning("PyPDF2 non disponible.")

    if not pdfplumber_available and not pypdf2_available:
        st.error("❌ Aucun parseur PDF installé. Ajoutez pdfplumber ou PyPDF2 dans requirements.txt.")
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

    with open(txt_path, "w", encoding="utf-8", errors="ignore") as f:
        f.write(full_text)


# ==========================================================
# 4) PRETRAITEMENT
# ==========================================================
def preprocess(sentence):
    words = word_tokenize(sentence)

    sw = set(stopwords.words("french"))
    punct = set(".,;:!?()[]{}'\"-–")

    words = [w.lower() for w in words if w.lower() not in sw and w not in punct]

    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]

    return words


# ==========================================================
# 5) SIMILARITE
# ==========================================================
def jaccard_similarity(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 0
    return len(a.intersection(b)) / len(a.union(b))


def find_best_sentence(query, sentences, corpus):
    query_tokens = preprocess(query)

    best_sim = 0
    best_sentence = "Aucune réponse trouvée dans le document."

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

    pdf_path = "Formation_Arche.pdf"
    txt_path = "formation_arche.txt"

    extract_pdf_to_txt(pdf_path, txt_path)

    if not os.path.exists(txt_path):
        st.error("❌ Le fichier texte n'a pas pu être généré.")
        st.stop()

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        raw_text = f.read()

    sentences = sent_tokenize(raw_text)
    corpus = [preprocess(s) for s in sentences]

    q = st.text_input("Posez votre question sur Arche Ossature :")

    if st.button("🔎 Rechercher"):
        if not q.strip():
            st.warning("Entrez une question valide.")
        else:
            st.markdown("### 📘 Réponse :")
            st.write(chatbot(q, sentences, corpus))


if __name__ == "__main__":
    main()
