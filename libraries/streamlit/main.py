import streamlit as st
import pandas as pd
from pathlib import Path

st.write("""
# My first app
Hello *world!*
""")

script_dir = Path(__file__).resolve().parent
df = pd.read_csv(script_dir / "my_data.csv", index_col=0)
st.line_chart(df)