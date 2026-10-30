import io
import fitz  # PyMuPDF
import pdfplumber
import pypdf

# Create a small valid PDF using PyMuPDF in memory
doc = fitz.open()
page = doc.new_page()
page.insert_text((50, 50), "(12) United States Patent\n(10) Patent No.: US 10,713,783 B2\n(45) Date of Patent: Jul. 14, 2020\n(54) NEURAL NETWORK CLASSIFICATION\n(22) Filed: Jun. 1, 2017\n(57) ABSTRACT\nNeural network classification may be performed by inputting a training data set.\nWhat is claimed is:\n1. A method for neural network classification comprising: inputting a training data set.\n5. A system for neural network classification comprising: a memory.\n7. A non-transitory computer-readable medium.")
pdf_bytes = doc.tobytes()

print(f"Generated test PDF bytes size: {len(pdf_bytes)} bytes")

def test_extract(file_bytes: bytes):
    print("\n1. Testing PyMuPDF fitz.open(stream=file_bytes, filetype='pdf')...")
    try:
        doc1 = fitz.open(stream=file_bytes, filetype="pdf")
        pages1 = [p.get_text("text") for p in doc1]
        print(f"PyMuPDF Success! Pages: {len(pages1)}, Text length: {len(pages1[0])}")
    except Exception as e:
        print(f"PyMuPDF Failed: {e}")

    print("\n2. Testing pdfplumber.open(io.BytesIO(file_bytes))...")
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf2:
            pages2 = [p.extract_text() for p in pdf2.pages]
            print(f"pdfplumber Success! Pages: {len(pages2)}, Text length: {len(pages2[0])}")
    except Exception as e:
        print(f"pdfplumber Failed: {e}")

    print("\n3. Testing pypdf.PdfReader(io.BytesIO(file_bytes))...")
    try:
        reader3 = pypdf.PdfReader(io.BytesIO(file_bytes))
        pages3 = [p.extract_text() for p in reader3.pages]
        print(f"pypdf Success! Pages: {len(pages3)}, Text length: {len(pages3[0])}")
    except Exception as e:
        print(f"pypdf Failed: {e}")

if __name__ == "__main__":
    test_extract(pdf_bytes)
