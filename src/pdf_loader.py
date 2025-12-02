from PyPDF2 import PdfReader

def load_pdf(file_path):
    all_text = ""
    try:
        pdf = PdfReader(file_path)

        if len(pdf.pages) == 0:
            raise ValueError("PDF has no pages")
        
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                all_text += page_text
        
        if not all_text.strip():
            raise ValueError("No text could be extracted from PDF")
        
        return all_text
        
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found : {file_path}")
    
    except Exception as e:
        raise Exception(f"Error reading PDF : {str(e)}")

if __name__ == "__main__":
    pdf_path = "data/sample_pdfs/test.pdf"
    text = load_pdf(pdf_path)

    if text:
        print(f"{len(text)} characters extracted from {pdf_path}")
        print("first 500 characters : ")
        print(text[:500])