from pypdf import PdfReader
import sys

print("--- START PDF EXTRACT ---")
try:
    reader = PdfReader("bct algo.pdf")
    for i, page in enumerate(reader.pages):
        print(f"--- PAGE {i+1} ---")
        print(page.extract_text())
except Exception as e:
    print("Error:", e)
print("--- END PDF EXTRACT ---")
