from fpdf import FPDF
import datetime

def export_answer_to_pdf(question, answer):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.cell(200, 10, txt="SpecBot Q&A Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=11)
    
    pdf.multi_cell(0, 10, f"Question:\n{question}")
    pdf.ln(5)
    pdf.multi_cell(0, 10, f"Answer:\n{answer}")
    pdf.ln(10)
    pdf.set_font("Arial", size=9)
    pdf.cell(0, 10, f"Exported on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)

    filepath = "specbot_answer.pdf"
    pdf.output(filepath)
    return filepath
