from PyPDF2 import PdfReader

def load_pdf(file_path):
    all_text = ""
    try:
        pdf = PdfReader(file_path)
        for page in pdf.pages:
            page_text = page.extract_text()
            all_text += page_text
        
        return all_text
        
    except FileNotFoundError:
        print(f"file not found at {file_path}")
        return ""
    
    except Exception as e:
        print(f"error reading pdf : {e}")
        return ""

if __name__ == "__main__":
    pdf_path = "data/sample_pdfs/test.pdf"
    text = load_pdf(pdf_path)

    if text:
        print(f"{len(text)} characters extracted from {pdf_path}")
        print("first 500 characters : ")
        print(text[:500])