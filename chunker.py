import pymupdf

def extract_text_from_pdf(pdf_path):
    text = ""

    doc = pymupdf.open(pdf_path)

    for page in doc:
        text += page.get_text()

    doc.close()
    return text


def chunk_text(text, chunk_size=1200, overlap=200):
    chunks = []

    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks