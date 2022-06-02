import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from  PIL import Image
import numpy as np
import pandas as pd
import plotly.figure_factory as ff

# 1. Menu
# 1.1 Sidebar menu
# with st.sidebar:
#     selected = option_menu(
#         menu_title = None,
#         options=['Home', 'Visualisations', 'Contact'],
#         icons=["house", "bar-chart-fill", "envelope"],
#         menu_icon="cast",
#         default_index=0,
#         # orientation="horizontal",
#         styles={
#         "container": {"padding": "5!important", "background-color": "#fafafa"},
#         "icon": {"color": "orange", "font-size": "25px"},
#         "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
#         "nav-link-selected": {"background-color": "#02ab21"},
#     },
#     )

# 1.2 Horizontal menu bar
selected = option_menu(
        menu_title = None,
        options=['Home', 'Visualisations', 'Contact'],
        icons=["house", "bar-chart-fill", "envelope"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
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
    st.write(f'**{input_keywords}**')

# random visualisations
if selected == "Visualisations":
    st.title(f"You have selected {selected}")
    # st.bar_chart()
    # Add histogram data
    x1 = np.random.randn(200) - 2
    x2 = np.random.randn(200)
    x3 = np.random.randn(200) + 2

    # Group data together
    hist_data = [x1, x2, x3]

    group_labels = ['Group 1', 'Group 2', 'Group 3']

    # Create distplot with custom bin_size
    fig = ff.create_distplot(
            hist_data, group_labels, bin_size=[.1, .25, .5])

    # Plot!
    st.plotly_chart(fig, use_container_width=True)

if selected == "Contact":
    st.title(f"You have selected {selected}")
    st.write("Creators: ")
    st.write("Axel Pichler")
    st.write("Jakob Gübel")
    st.write("Felix van Litsenburg")
    st.write("Christopher Peter")
