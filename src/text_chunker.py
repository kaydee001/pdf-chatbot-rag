def find_split_point(text, ideal_position, search_range=100):
    start = max(0, ideal_position - search_range)
    end = min(len(text), ideal_position + search_range)

    segment = text[start:end]
    last_period = segment.rfind(". ")
    
    if last_period != -1:
        return start+last_period+2
    else:
        return ideal_position

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0

    while start < len(text):
        # end = start + chunk_size
        end = find_split_point(text, start + chunk_size)
        chunk = text[start:end]
        chunks.append(chunk)
        start = start + (chunk_size-overlap)

    return chunks

if __name__ == "__main__":
    from pdf_loader import load_pdf

    pdf_path = "data/sample_pdfs/test.pdf"
    text = load_pdf(pdf_path)

    chunks = chunk_text(text, chunk_size=1000, overlap=200)

    print(f"total length : {len(text)} characters")
    print(f"no of chunks : {len(chunks)}")
    print(f"first chunk size : {len(chunks[0])}")
    print(f"last chunk size : {len(chunks[-1])}")
