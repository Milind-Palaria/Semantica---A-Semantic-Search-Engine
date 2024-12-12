import pandas as pd
from io import BytesIO
import xlsxwriter

def generate_csv(dataframe):
    return dataframe.to_csv(index=False).encode('utf-8')

def generate_excel(dataframe):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    dataframe.to_excel(writer, sheet_name='Sheet1', index=False)
    writer.close()
    return output.getvalue()