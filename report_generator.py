import pandas as pd
from io import BytesIO

def generate_csv(dataframe):
    return dataframe.to_csv(index=False).encode('utf-8')
