"""
Temporary script to extract text from a Word document to markdown.
"""
import sys
import subprocess

# Check if python-docx is installed, if not install it
try:
    import docx
except ImportError:
    print("python-docx not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-docx"])
    import docx

# Extract text from docx
docx_path = r"c:\Users\16179\Desktop\ARIMA\ref\doc\初稿中的新.docx"
output_path = r"c:\Users\16179\Desktop\ARIMA\ref\doc\thesis_summary.md"

print(f"Reading from: {docx_path}")
doc = docx.Document(docx_path)

# Extract all paragraphs
paragraphs = []
for para in doc.paragraphs:
    text = para.text.strip()
    if text:  # Only add non-empty paragraphs
        paragraphs.append(text)

# Join paragraphs with newlines
content = "\n".join(paragraphs)

# Save to markdown file
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Successfully extracted text to: {output_path}")
print(f"Total paragraphs: {len(paragraphs)}")
