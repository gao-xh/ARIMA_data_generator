from docx import Document
import sys

filename = sys.argv[1]
try:
    doc = Document(filename)
    fullText = []
    for paragraph in doc.paragraphs:
        fullText.append(paragraph.text)
    print("\n".join(fullText))
except Exception as e:
    print(f"Error reading file: {e}")
