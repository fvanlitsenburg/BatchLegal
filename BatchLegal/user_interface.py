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

from BatchLegal.visualization_descriptive import *

st.set_page_config(
    page_title = "BatchLegal",
    page_icon="⚖️",
    layout="wide"
)
#  title
st.title("BatchLegal")
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
            ### With the help of the following 2️⃣ topic modelling algorithms we are able to grasp the most important topics of legal texts.
            ''')
    st.markdown('''
            ## LDA: Latent Dirichlet Allocation ❓\n
            🔘 "Latent" = hidden (topics)  \n
            🔘 "Dirichlet" = type of probability distribution  \n
            🔘 Unsupervised algorithm to find topics in documents
            ''')
    st.markdown('''
            ## BERTopic ❓ \n
            🔘 leverages transformers and c-TF-IDF to create dense clusters  \n
            🔘 allows for easily interpretable topics  \n
            🔘 keeping important words in the topic descriptions.  \n
            # 🏁
            ''')

    input_keywords = st.text_input("", placeholder = "Please type in your keyword...")
    st.write(f'**{input_keywords}**')

# descriptive visualization of metadata
if selected == "Visualisations":
    filename = "../raw_data/20220602.csv"
    data = load_metadata_for_vis(filename)
    
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
    '''
    fig_col1, fig_col2 = st.columns(2)
    with fig_col1:
        st.plotly_chart(fig1)
    with fig_col2:
        st.plotly_chart(fig2)
    '''
    fig3 = visualization_stackedarea_normalized(data_subset_time, plottype="plotly")
    st.plotly_chart(fig3)



import itertools
from typing import List
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if selected == "Model Output":

    def bert_bar(topic_freq, get_topic,
                topics: List[int] = None,
                top_n_topics: int = 10,
                n_words: int = 5,
                width: int = 250,
                height: int = 250) -> go.Figure:
        colors = itertools.cycle(["#D55E00", "#0072B2", "#CC79A7", "#E69F00", "#56B4E9", "#009E73", "#F0E442"])
        # Select topics based on top_n and topics args
        freq_df = topic_freq
        #freq_df = freq_df.loc[freq_df.Topic != -1, :]
        if topics is not None:
            topics = list(topics)
        elif top_n_topics is not None:
            topics = sorted(freq_df.Topic.to_list()[:top_n_topics])
        else:
            topics = sorted(freq_df.Topic.to_list()[0:6])
        print(topics)
        # Initialize figure
        subplot_titles = [f"Topic {topic}" for topic in topics]
        columns = 4
        rows = int(np.ceil(len(topics) / columns))
        fig = make_subplots(rows=rows,
                        cols=columns,
                        shared_xaxes=False,
                        horizontal_spacing=.1,
                        vertical_spacing=.4 / rows if rows > 1 else 0,
                        subplot_titles=subplot_titles)
        # Add barchart for each topic
        row = 1
        column = 1
        for topic in topics:
            words = [word + "  " for word, _ in get_topic[topic]][:n_words][::-1]
            scores = [score for _, score in get_topic[topic]][:n_words][::-1]
            fig.add_trace(
                go.Bar(x=scores,
                        y=words,
                        orientation='h',
                        marker_color=next(colors)),
                row=row, col=column)
            if column == columns:
                column = 1
                row += 1
            else:
                column += 1
        # Stylize graph
        fig.update_layout(
            template="plotly_white",
            showlegend=False,
            title={
                'text': "<b>Topic Word Scores",
                'x': .5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(
                    size=22,
                    color="Black")
            },
            width=width*4,
            height=height*rows if rows > 1 else height * 1.3,
            hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
            ),
        )
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True)
        return fig

    data_axel = pd.read_pickle("../raw_data/sub_dir_topics_axel.pkl")
    topic_freq = data_axel['get_topic_freq']
    get_topic = data_axel['get_topics']
    # pick topic
    topiclist = data_axel['Sub_dir Name:'].tolist()
    chosen_topic = st.selectbox('Select subdirectory:', topiclist)
    st.plotly_chart(bert_bar(topic_freq[topiclist.index(chosen_topic)], get_topic[topiclist.index(chosen_topic)]))


if selected == "Contact":
    st.title(f"You have selected {selected}")
    st.write("Creators: Axel Pichler, Jakob Gübel, Felix van Litsenburg, Christopher Peter")
    # st.write("Axel Pichler")
    # st.write("Jakob Gübel")
    # st.write("Felix van Litsenburg")
    # st.write("Christopher Peter")
