import pandas as pd
from fpdf import FPDF

# Convert CSV to PDF
def convert_csv_to_pdf(csv_path, output_pdf):
    df = pd.read_csv(csv_path)
    pdf = FPDF()
    pdf.add_page()
    
    # Add table to PDF (simplified)
    for index, row in df.iterrows():
        pdf.set_font("Arial", size=12)
        line = ', '.join([str(elem) for elem in row])
        pdf.cell(200, 10, txt=line, ln=True)
    
    pdf.output(output_pdf)

# Example usage
convert_csv_to_pdf('data/statsfinal.csv', 'output_file.pdf')