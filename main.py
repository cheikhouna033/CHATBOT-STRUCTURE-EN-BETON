import PyPDF2

# Ouvrir le fichier PDF
pdf_file = open("STRUCTURE.pdf", "rb")
reader = PyPDF2.PdfReader(pdf_file)

full_text = ""

# Extraire chaque page
for page in reader.pages:
    text = page.extract_text()
    if text:
        full_text += text + "\n"

pdf_file.close()

# Sauvegarder dans un fichier texte
with open("structure.txt", "w", encoding="utf-8") as f:
    f.write(full_text)

print("Extraction terminée ! Ton fichier structure.txt est prêt.")
