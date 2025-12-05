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

# ... remainder of the file unchanged ...