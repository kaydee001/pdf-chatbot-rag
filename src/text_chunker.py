def find_split_point(text, ideal_position, search_range=100):
    # search window around ideal position
    start = max(0, ideal_position - search_range)
    end = min(len(text), ideal_position + search_range)

    # extracting the segment to search
    segment = text[start:end]

    # find last period in this segment
    last_period = segment.rfind(". ") # reverse find
    
    if last_period != -1: # period found
        # position after period -> start + offset + 2 for ". "
        return start+last_period+2
    else:
        # no period found
        return ideal_position

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0

    while start < len(text):
        # smart split point at sentence boundary (if possible)
        end = find_split_point(text, start + chunk_size)
        #extracting chunks
        chunk = text[start:end]
        chunks.append(chunk)
        # move start forward -> step size = chunk_size - overlap
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
