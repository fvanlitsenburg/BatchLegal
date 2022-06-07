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

st.set_page_config(
    page_title = "BatchLegal",
    page_icon="⚖️",
    layout="wide"
)
#  title
st.title("BatchLegal")

# 1.2 Horizontal menu bar
selected = option_menu(
        menu_title = None,
        options=['Home', 'Visualisations', 'Model Output','Contact'],
        icons=["house", "bar-chart-fill","bar-chart-fill", "envelope"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
        "container": {"width": "900px", "padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"},
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    },
    )

if selected == "Home":
    # st.title(f"You have selected {selected} directory.")
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
if selected == "Visualisations":
    url = "https://drive.google.com/file/d/1IUcEktP1RHDnnTcl4M5_QIL2sSNwVW3r/view?usp=sharing"
    path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
    data = pd.read_csv(path)
    data['date'] = pd.to_datetime(data['date'])
    data = data[~data["dir_1"].isna()].reset_index().drop(columns = "index") # drop rows that have NA in dir_1 column

    # call function to display the top layer directories
    dir_1 = exploration_list_subdirs(data, dir_1=None, dir_2=None)
    # user input directory-level and keyword
    columns = st.columns(2)
    dir_1_selection = columns[0].selectbox('Select main directory', ["No Selection"] + dir_1)
    # call function again based on input
    dir_2 = exploration_list_subdirs(data, dir_1=dir_1_selection, dir_2=None)
    # user input for second layer directories
    dir_2_selection = columns[1].selectbox('Select sub-directory', ["No Selection"] + dir_2)

    # select timeframe and sampling method
    columns = st.columns(3)
    start_date = columns[0].date_input("Start date 🗓:", datetime.date(2011, 1, 1))
    # columns[0].write(start_date)
    end_date = columns[1].date_input("End date 📆:", datetime.date(2022, 12, 31))
    # columns[1].write(end_date)
    time_selection = columns[2].selectbox('Per month or per year? ⌛️', ['Year', 'Month'])
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


    fig1 = visualization_piechart(data_subset_time)
    st.plotly_chart(fig1)

    fig2 = visualization_stackedarea(data_subset_time, plottype="plotly")
    st.plotly_chart(fig2)

    fig3 = visualization_stackedarea_normalized(data_subset_time, plottype="plotly")
    st.plotly_chart(fig3)



import itertools
from typing import List
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from BatchLegal.bert_viz import *

if selected == "Model Output":
    url_top = "https://drive.google.com/file/d/1DLVflFOzf30kOKzNiktVSCrCnY5EmlNm/view?usp=sharing"
    path_top = 'https://drive.google.com/uc?export=download&id='+url_top.split('/')[-2]
    topics_dir1_df = pd.read_pickle(path_top)
    url_emb = "https://drive.google.com/file/d/1dG5nxr_lLuFjjLtYu8hki9Im6PK91XR-/view?usp=sharing"
    path_emb = "https://drive.google.com/uc?export=download&id="+url_emb.split('/')[-2]
    embeddings_lst = pd.read_pickle(path_emb)
    # with open(path_emb, 'rb') as handle:
    #     embeddings_lst = pickle.load(handle)

    url_dist = "https://drive.google.com/file/d/1uAQKY7ovlqGRl1Zobd_GNdrXPR9aQkZO/view?usp=sharing"
    path_dist = "https://drive.google.com/uc?export=download&id="+url_dist.split('/')[-2]
    distances_lst = pd.read_pickle(path_dist)
    # with open(path_dist, 'rb') as handle:
    #     distances_lst = pickle.load(handle)


    topic_list = topics_dir1_df['Sub_dir Name:'].tolist()
    # pick topic
    theme = st.selectbox('Select subdirectory:', topic_list)

    temp = theme.split(' ')
    temp = temp[0].replace(',', '')
    embeds = temp+'_embeds'
    dist = temp+'_dist'

    topic_list_index = topic_list.index(theme)
    topic_freq = topics_dir1_df.iloc[topic_list_index]['topic_freq']
    get_topic = topics_dir1_df.iloc[topic_list_index]['get_topic']
    topic_sizes = topics_dir1_df.iloc[topic_list_index]['topic_sizes']



    st.plotly_chart(bert_bar(topic_freq, get_topic))


    if len(embeddings_lst[topic_list_index][embeds]) == 0:
        st.write("Not enough topics to visualize")
    else:
        st.plotly_chart(visualize_topics(topic_freq, topic_sizes, get_topic, embeddings_lst[topic_list_index][embeds]))
        st.plotly_chart(visualize_hierarchy(topic_freq, get_topic, distances_lst[topic_list_index][dist]))


if selected == "Contact":
    st.title(selected)
    st.write("Creators:  \n Axel Pichler  \n Jakob Gübel  \n Felix van Litsenburg  \n Christopher Peter")

# end
