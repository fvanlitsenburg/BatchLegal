import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from  PIL import Image
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import datetime
import pickle
from BatchLegal.visualization_descriptive import *

## Load data
@st.cache(allow_output_mutation=True)
def get_data():
    url = "https://drive.google.com/file/d/1IUcEktP1RHDnnTcl4M5_QIL2sSNwVW3r/view?usp=sharing"
    path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
    data = pd.read_csv(path)
    return data

# url_top = "https://drive.google.com/file/d/1DLVflFOzf30kOKzNiktVSCrCnY5EmlNm/view?usp=sharing"
@st.cache(allow_output_mutation=True)
def get_topic(dir = 1):
    if dir == 1:
        url_top = "https://drive.google.com/file/d/1DLVflFOzf30kOKzNiktVSCrCnY5EmlNm/view?usp=sharing"
    elif dir == 2:
        url_top = "https://drive.google.com/file/d/1zL5vEA77S4ugZP2HNY69N4vdroqd3XRJ/view?usp=sharing"
    elif dir == 3:
        url_top = "https://drive.google.com/file/d/1k3dLSHmG2BzuBcZRL1a_MheiIiXl5Ysa/view?usp=sharing"
    path_top = 'https://drive.google.com/uc?export=download&id='+url_top.split('/')[-2]
    topics_dir_df = pd.read_pickle(path_top)
    return topics_dir_df

@st.cache(allow_output_mutation=True)
def get_emb(dir = 1):
    if dir == 1:
        url_emb = "https://drive.google.com/file/d/1dG5nxr_lLuFjjLtYu8hki9Im6PK91XR-/view?usp=sharing"
    elif dir == 2:
        url_emb = "https://drive.google.com/file/d/1FFMMaqvrSkiGX-2HgM08iVSt6ZkzAafv/view?usp=sharing"
    elif dir == 3:
        url_emb = "https://drive.google.com/file/d/1vwawZUsO9TqYCQSeXiMyHuVDh4E2Y6Tz/view?usp=sharing"
    path_emb = 'https://drive.google.com/uc?export=download&id='+url_emb.split('/')[-2]
    embeddings_lst = pd.read_pickle(path_emb)
    return embeddings_lst

@st.cache(allow_output_mutation=True)
def get_dist(dir = 1):
    if dir == 1:
        url_dist = "https://drive.google.com/file/d/1uAQKY7ovlqGRl1Zobd_GNdrXPR9aQkZO/view?usp=sharing"
    elif dir == 2:
        url_dist = "https://drive.google.com/file/d/1LOub6h0nnmOxMM8aIrWZGdnIbO3-JmjJ/view?usp=sharing"
    elif dir == 3:
        url_dist = "https://drive.google.com/file/d/1HLWilwWEf_mCq5S8VgGTgUQmpKcm0T6f/view?usp=sharing"
    path_dist = "https://drive.google.com/uc?export=download&id="+url_dist.split('/')[-2]
    distances_lst = pd.read_pickle(path_dist)
    return distances_lst

st.set_page_config(
    page_title = "BatchLegal",
    page_icon="⚖️",
    layout="wide"
)
#  title
col1, col2, col3 = st.columns(3)
with col2:
    st.title("BatchLegal")

# 1.2 Horizontal menu bar
selected = option_menu(
        menu_title = None,
        options=['Home', 'EU Data Overview', 'Topic Modelling','Contact'],
        icons=["house", "bar-chart-fill","bar-chart-fill", "envelope"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
        "container": {"width": "950px", "padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"},
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    },
    )

if selected == "Home":
    # st.title(f"You have selected {selected} directory.")
    st.sidebar.empty()
    st.markdown('''
            # Wondering what is going within EU regulations?  \n
            ## Our App **BatchLegal** will give you an overview! _Cool_? ✅  \n
            With the help of the following 2️⃣ topic modelling algorithms we are able to grasp the most important topics of legal texts.
            ''')
    st.markdown('''
            ## BERTopic ❓ \n
            🔘 leverages transformers and c-TF-IDF to create dense clusters  \n
            🔘 allows for easily interpretable topics  \n
            🔘 keeping important words in the topic descriptions.  \n
            # 🏁
            ''')

# descriptive visualization of metadata
if selected == "EU Data Overview":
    data = get_data()
    data['date'] = pd.to_datetime(data['date'])
    data = data[~data["dir_1"].isna()].reset_index().drop(columns = "index") # drop rows that have NA in dir_1 column

    # call function to display the top layer directories
    dir_1 = exploration_list_subdirs(data, dir_1=None, dir_2=None)
    # user input directory-level and keyword

    dir_1_selection = st.sidebar.selectbox('Select main directory 📜', ["No Selection"] + dir_1)
    # call function again based on input
    dir_2 = exploration_list_subdirs(data, dir_1=dir_1_selection, dir_2=None)
    # user input for second layer directories
    dir_2_selection = st.sidebar.selectbox('Select sub-directory 📜', ["No Selection"] + dir_2)

    # select timeframe and sampling method
    start_date = st.sidebar.date_input("Start date 🗓:", datetime.date(2011, 1, 1))
    # columns[0].write(start_date)
    end_date = st.sidebar.date_input("End date 📆:", datetime.date(2022, 12, 31))
    # columns[1].write(end_date)
    time_selection = st.sidebar.selectbox('Time sampling by month or by year? ⌛️', ['Year', 'Month'])
    # columns[2].write(location)
    if time_selection == 'Year':
        timesampling = "Y"
    else:
        timesampling = "M"

    # case when no directories are selected:
    if np.logical_and(dir_1_selection == "No Selection", dir_2_selection == "No Selection"):
        data_subset, dirlevel = subset_data_subdirs(data, dir_1=None, dir_2=None, dir_3=None)
    # case when only dir 1 selected:
    elif np.logical_and(dir_1_selection != "No Selection", dir_2_selection == "No Selection"):
        data_subset, dirlevel = subset_data_subdirs(data, dir_1=dir_1_selection, dir_2=None, dir_3=None)
    # case when dir 1 and dir 2 selected:
    elif np.logical_and(dir_1_selection != "No Selection", dir_2_selection != "No Selection"):
        data_subset, dirlevel = subset_data_subdirs(data, dir_1=dir_1_selection, dir_2=dir_2_selection, dir_3=None)

    # subsetting for time
    data_subset_time = subset_data(data_subset, start_date=str(start_date), end_date=str(end_date), timesampling=timesampling, directory_level=dirlevel)

    try:
        fig1 = visualization_barchart(data_subset_time)
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = visualization_stackedarea(data_subset_time, plottype="plotly")
        st.plotly_chart(fig2, use_container_width=True)

        fig3 = visualization_stackedarea_normalized(data_subset_time, plottype="plotly")
        st.plotly_chart(fig3, use_container_width=True)
    except:
        st.write("**Not enough data to visualise here.**")



import itertools
from typing import List
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from BatchLegal.bert_viz import *

if selected == "Topic Modelling":
    dir_list = ['- Agriculture', '--- Fresh fruit and vegetables', '--- Milk products', '--- Oils and fats', '--- Sugar', '--- Wine', '-- Agricultural structural funds', '--- European Agricultural Guarantee Fund', '--- Social and structural measures', '-- Statistics', '--- Arrangements covering more than one market organisation', '* Area of freedom, security and justice', '--- Crossing external borders', '--- Origin of goods', '-- Police and judicial cooperation in criminal and customs matters', '-- Programmes', '- Competition policy', '- Customs Union and free movement of goods', '-- Application of the Common Customs Tariff', '--- Tariff classification', '-- General', '-- General customs rules', '--- Common customs territory', '* Economic and monetary policy and free movement of capital', '--- Institutional economic provisions', '--- Instruments of economic policy', '* Energy', '-- Electricity', '-- General principles and programmes', '--- Rational utilisation and conservation of energy', '- Environment, consumers and health protection', '--- Pollution and nuisances', '--- Space, environment and natural resources', '-- Consumers', '--- Consumer information, education and representation', '--- Protection of economic interests', '-- Health protection', '- External relations', '-- Bilateral agreements with non-member countries ', '-- Commercial policy', '--- Other commercial policy measures', '--- Trade arrangements', '-- Development policy', '--- Generalised system of preferences', '-- External relations', '* Fisheries', '--- Agreements with non-member countries', '--- Market organisation', '--- Structural measures', '* Freedom of movement for workers and social policy', '-- Social policy', '--- General social provisions', '- General, financial and institutional matters', '-- Financial and budgetary provisions', '-- Provisions governing the institutions', '- Industrial policy and internal market', '-- Industrial policy: general, programmes, statistics and research', '-- Industrial policy: sectoral operations', '--- Information technology, telecommunications and data-processing', '-- Internal market: approximation of laws', '--- Agricultural and forestry tractors', '--- Dangerous substances', '--- Motor vehicles', '--- Plant health', '--- Proprietary medicinal products', '* Law relating to undertakings', '-- Judicial cooperation in civil matters', '- Regional policy and coordination of structural instruments', '-- Coordination of structural instruments', '-- Dissemination of information', '- Taxation', '- Transport policy', '--- Market operation', '-- Air transport', '--- Air safety', '-- Shipping']
    theme = st.sidebar.selectbox('Select directory:', dir_list)

    if theme[0:4] == '--- ':
        theme = theme[4:]
        dir = 3
        validtheme = True
    elif theme[0:3] == '-- ':
        theme = theme[3:]
        dir = 2
        validtheme = True
    elif theme[0:2] == '* ':
        validtheme = False
    else:
        theme = theme[2:]
        dir = 1
        validtheme = True

    if validtheme == True:

        temp = theme.split(' ')
        temp = temp[0].replace(',', '')
        embeds = temp+'_embeds'
        dist = temp+'_dist'

        topics_dir1_df = get_topic(dir)
        topic_list = topics_dir1_df['Sub_dir_name'].tolist()

        topic_list_index = topic_list.index(theme)
        topic_freq = topics_dir1_df.iloc[topic_list_index]['topic_freq']
        get_topic = topics_dir1_df.iloc[topic_list_index]['get_topic']
        topic_sizes = topics_dir1_df.iloc[topic_list_index]['topic_sizes']



        st.plotly_chart(bert_bar(topic_freq, get_topic), use_container_width=True)

        embeddings_lst = get_emb(dir)
        distances_lst = get_dist(dir)
        if len(embeddings_lst[topic_list_index][embeds]) == 0:
            st.write("Not enough topics to visualize")
        else:
            st.plotly_chart(visualize_topics(topic_freq, topic_sizes, get_topic, embeddings_lst[topic_list_index][embeds]), use_container_width=True)
            st.plotly_chart(visualize_hierarchy(topic_freq, get_topic, distances_lst[topic_list_index][dist]), use_container_width=True)
    else:
        st.write("Not enough data to visualise. Please select another (sub)directory.")


if selected == "Contact":
    st.sidebar.empty()
    st.title(selected)
    st.write("Creators:  \n Axel Pichler  \n Jakob Gübel  \n Felix van Litsenburg  \n Christopher Peter")

# end
