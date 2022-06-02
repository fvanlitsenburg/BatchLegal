import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from  PIL import Image
import numpy as np
import pandas as pd

# 1. sidebar menu

with st.sidebar:
    selected = option_menu(
        menu_title = None,
        options=['Home', 'Visualisations', 'Contact'],
        icons=["house", "bar-chart-fill", "envelope"],
        menu_icon="cast",
        default_index=0,
        # orientation="horizontal",
        styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"},
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    },
    )

if selected == "Home":
    # st.title(f"You have selected {selected} directory.")
    st.markdown('''
            Wondering what is going with EU regulations?
            **BatchLegal** will give you an overview! _Cool_?
            ''')
    input_keywords = st.text_input("", placeholder = "Please type in your keyword...")
    st.write(f'**_{input_keywords}_**')

if selected == "Visualisations":
    st.title(f"You have selected {selected}")
    # st.bar_chart()

if selected == "Contact":
    st.title(f"You have selected {selected}")
    st.write("Creators: ")
    st.write("Axel Pichler")
    st.write("Jakob Gübel")
    st.write("Felix van Litsenburg")
    st.write("Christopher Peter")
