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

# random visualisations
if selected == "Visualisations":
    filename = "../raw_data/20220602.csv"
    data = pd.read_csv(filename).drop(columns = {'Unnamed: 0'})
    data['date'] = pd.to_datetime(data['date'])
    data = data[~data['dir_1'].isna()].reset_index().drop(columns = "index")

    columns = st.columns(3)

    d = columns[0].date_input("Start date 🗓:", datetime.date(2011, 1, 1))
    # columns[0].write(start_date)

    e = columns[1].date_input("End date 📆:", datetime.date(2022, 12, 31))
    # columns[1].write(end_date)

    time_selection = columns[2].selectbox('Per month or per year? ⌛️', ['Year', 'Month'])
    # columns[2].write(location)

    if time_selection == 'Year':
        timesampling = "Y"
    else:
        timesampling = "M"


    start_date = datetime.datetime.strptime(str(d), '%Y-%m-%d')
    end_date = datetime.datetime.strptime(str(e), '%Y-%m-%d')
    data_subset = data[np.logical_and(data['date'] >= start_date, data['date'] <= end_date)]
    # uncomment this line if no time-subset is wanted
    #data_subset = data
    # select the sampling method

    dir_1 = data['dir_1'].value_counts().index
    df = data_subset[data_subset['dir_1'] == dir_1[0]].resample(timesampling, on='date')['title'].count().reset_index().rename(columns={'title':dir_1[0]})
    for i in range(1,len(dir_1)):
        category = dir_1[i]
        temp = data_subset[data_subset['dir_1'] == category].resample(timesampling, on='date')['title'].count().reset_index().rename(columns={'title':category})
        df = df.merge(temp, how='left', on='date').fillna(0)
    data_publications = pd.concat([df['date'], df.drop(columns = "date").astype('Int64')], axis=1)

    piedata = data_publications.drop(columns='date').sum().reset_index()
    fig1 = px.pie(piedata, values=0, names='index', title='Directories of published documents')
    #fig.update_layout(hovermode="x")
    # st.plotly_chart(fig)



    # prepare data
    x = data_publications['date'].tolist()
    y = data_publications.drop(columns = {"date"}).T.values.tolist()
    labels = data_publications.drop(columns = {"date"})
    # create dict for the labels in plotly
    newnames = {}
    for index in range(0,len(labels.columns)):
        newnames[f"wide_variable_{str(index)}"] = labels.columns[index]
    # matplotlib
    fig2 = plt.figure(figsize=(12,7))
    plt.stackplot(x,y, labels=labels)
    plt.legend()
    plt.xlabel("Date of Publication")
    plt.ylabel("Number of Publications")
    plt.title(f"Publication of EU-Regulations per Topic (stacked)")
    plt.show()
    # plotly
    x_plot = x.copy()
    y_plot = y.copy()
    fig2 = px.area(x=x_plot, y=y_plot,
                labels={"x": "Date of Publication",
                        "value": "Number of Publications",
                        "variable": "Category"},
                title='Publication of EU-Regulations per Topic (stacked)')
    fig2.for_each_trace(lambda t: t.update(name = newnames[t.name],
                                        legendgroup = newnames[t.name],
                                        hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name])))
    # st.plotly_chart(fig)

    fig_col1, fig_col2 = st.columns(2)

    with fig_col1:
        st.plotly_chart(fig1)

    with fig_col2:
        st.plotly_chart(fig2)

    # prepare data
#normalize
    df = data_publications.drop(columns = {'date'})
    data_publications_normalized = df.div(df.sum(axis=1), axis=0)
    y_norm = data_publications_normalized.T.values.tolist()
    x_norm = x.copy() # see chapter above
    labels = labels.copy() # see chapter above
    newnames = newnames.copy() # dict for labels in plotly, see chapter above
    # plotly
    x_norm_plot = x_norm.copy()
    y_norm_plot = y_norm.copy()
    fig = px.area(x=x_norm_plot, y=y_norm_plot,
                labels={"x": "Date of Publication",
                        "value": "Share of Publications in this Topic",
                        "variable": "Category"},
                title='Publication of EU-Regulations per Topic (stacked and normalized)')
    fig.for_each_trace(lambda t: t.update(name = newnames[t.name],
                                        legendgroup = newnames[t.name],
                                        hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name])))
    st.plotly_chart(fig)



import itertools
import numpy as np
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
