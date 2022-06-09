import itertools
import numpy as np
import pandas as pd
from typing import List

from scipy.cluster.hierarchy import linkage
from sklearn.metrics.pairwise import cosine_similarity

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

def bert_bar(topic_freq, get_topic,
             topics: List[int] = None,
             top_n_topics: int = None,
             n_words: int = 8,
             width: int = 250,
             height: int = 250) -> go.Figure:

  """Function takes the values of the 'topic_freq' and the 'get_topic' column of the Topics-Dataframes and returns a bar-chart of the top-5 words of each topic"""

  colors = itertools.cycle(["#D55E00", "#0072B2", "#CC79A7", "#E69F00", "#56B4E9", "#009E73", "#F0E442"])

  # Select topics based on top_n and topics args
  freq_df = topic_freq
  #freq_df = freq_df.loc[freq_df.Topic != -1, :]
  if topics is not None:
      topics = list(topics)
  elif top_n_topics is not None:
      topics = sorted(freq_df.Topic.to_list()[:top_n_topics])
  else:
      topics = sorted(freq_df.Topic.to_list())

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


def visualize_topics(topic_freq, topic_sizes, get_topic, embeddings,
                     topics: List[int] = None,
                     top_n_topics: int = None,
                     width: int = 650,
                     height: int = 650) -> go.Figure:
    """ Visualize topics, their sizes, and their corresponding words

    This visualization is highly inspired by LDAvis, a great visualization
    technique typically reserved for LDA.

    Arguments:
        values of the 'topic_freq', 'topic_sizes' and 'get_topic'-columns of the Topics-Dataframes
        Corresponding array from the Embeddings-DF
    """
    # Select topics based on top_n and topics args
    freq_df = topic_freq
    #freq_df = freq_df.loc[freq_df.Topic != -1, :]
    if topics is not None:
        topics = list(topics)
    elif top_n_topics is not None:
        topics = sorted(freq_df.Topic.to_list()[:top_n_topics])
    else:
        topics = sorted(freq_df.Topic.to_list())

    # Extract topic words and their frequencies
    topic_list = sorted(topics)
    frequencies = [topic_sizes[topic] for topic in topic_list]
    words = [" | ".join([word[0] for word in get_topic[topic][:5]]) for topic in topic_list]

    # Visualize with plotly
    df = pd.DataFrame({"x": embeddings[:, 0], "y": embeddings[:, 1],
                       "Topic": topic_list, "Words": words, "Size": frequencies})
    return _plotly_topic_visualization(df, topic_list, width, height)


def _plotly_topic_visualization(df: pd.DataFrame,
                                topic_list: List[str],
                                width: int,
                                height: int):
    """ Create plotly-based visualization of topics with a slider for topic selection """

    def get_color(topic_selected):
        if topic_selected == -1:
            marker_color = ["#B0BEC5" for _ in topic_list]
        else:
            marker_color = ["red" if topic == topic_selected else "#B0BEC5" for topic in topic_list]
        return [{'marker.color': [marker_color]}]

    # Prepare figure range
    x_range = (df.x.min() - abs((df.x.min()) * .15), df.x.max() + abs((df.x.max()) * .15))
    y_range = (df.y.min() - abs((df.y.min()) * .15), df.y.max() + abs((df.y.max()) * .15))

    # Plot topics
    fig = px.scatter(df, x="x", y="y", size="Size", size_max=40, template="simple_white", labels={"x": "", "y": ""},
                     hover_data={"Topic": True, "Words": True, "Size": True, "x": False, "y": False})
    fig.update_traces(marker=dict(color="#B0BEC5", line=dict(width=2, color='DarkSlateGrey')))

    # Update hover order
    fig.update_traces(hovertemplate="<br>".join(["<b>Topic %{customdata[0]}</b>",
                                                 "Words: %{customdata[1]}",
                                                 "Size: %{customdata[2]}"]))

    # Create a slider for topic selection
    steps = [dict(label=f"Topic {topic}", method="update", args=get_color(topic)) for topic in topic_list]
    sliders = [dict(active=0, pad={"t": 50}, steps=steps)]

    # Stylize layout
    fig.update_layout(
        title={
            'text': "<b>Intertopic Distance Map",
            'y': .95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        },
        width=width,
        height=height,
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
        xaxis={"visible": False},
        yaxis={"visible": False},
        sliders=sliders
    )

    # Update axes ranges
    fig.update_xaxes(range=x_range)
    fig.update_yaxes(range=y_range)

    # Add grid in a 'plus' shape
    fig.add_shape(type="line",
                  x0=sum(x_range) / 2, y0=y_range[0], x1=sum(x_range) / 2, y1=y_range[1],
                  line=dict(color="#CFD8DC", width=2))
    fig.add_shape(type="line",
                  x0=x_range[0], y0=sum(y_range) / 2, x1=x_range[1], y1=sum(y_range) / 2,
                  line=dict(color="#9E9E9E", width=2))
    fig.add_annotation(x=x_range[0], y=sum(y_range) / 2, text="D1", showarrow=False, yshift=10)
    fig.add_annotation(y=y_range[1], x=sum(x_range) / 2, text="D2", showarrow=False, xshift=10)
    fig.data = fig.data[::-1]

    return fig


def visualize_hierarchy(topic_freq, get_topic, distance_matrix,
                        orientation: str = "left",
                        topics: List[int] = None,
                        top_n_topics: int = None,
                        width: int = 1000,
                        height: int = 600) -> go.Figure:
    """ Visualize a hierarchical structure of the topics

    A ward linkage function is used to perform the
    hierarchical clustering based on the cosine distance
    matrix between topic embeddings.

    Arguments:
        values of the 'topic_freq', and 'get_topic'-columns of the Topics-Dataframes
        Corresponding array from the Distances-DF

    Returns:
        fig: A plotly figure
    """

    # Select topics based on top_n and topics args
    freq_df = topic_freq
    #freq_df = freq_df.loc[freq_df.Topic != -1, :]
    if topics is not None:
        topics = list(topics)
    elif top_n_topics is not None:
        topics = sorted(freq_df.Topic.to_list()[:top_n_topics])
    else:
        topics = sorted(freq_df.Topic.to_list())

    # Select embeddings
    #all_topics = sorted(list(get_topic.keys()))
    #indices = np.array([all_topics.index(topic) for topic in topics])
    #embeddings = embeds[indices]

    # Create dendogram
    distance_matrix = distance_matrix
    fig = ff.create_dendrogram(distance_matrix,
                               orientation=orientation,
                               linkagefun=lambda x: linkage(x, "ward"),
                               color_threshold=1)

    # Create nicer labels
    axis = "yaxis" if orientation == "left" else "xaxis"
    new_labels = [[[str(topics[int(x)]), None]] + get_topic[topics[int(x)]]
                  for x in fig.layout[axis]["ticktext"]]
    new_labels = ["_".join([label[0] for label in labels[:4]]) for labels in new_labels]
    new_labels = [label if len(label) < 30 else label[:27] + "..." for label in new_labels]

    # Stylize layout
    fig.update_layout(
        plot_bgcolor='#ECEFF1',
        template="plotly_white",
        title={
            'text': "<b>Hierarchical Clustering",
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        },
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
    )

    # Stylize orientation
    if orientation == "left":
        fig.update_layout(height=200+(15*len(topics)),
                          width=width,
                          yaxis=dict(tickmode="array",
                                     ticktext=new_labels))

        # Fix empty space on the bottom of the graph
        y_max = max([trace['y'].max()+5 for trace in fig['data']])
        y_min = min([trace['y'].min()-5 for trace in fig['data']])
        fig.update_layout(yaxis=dict(range=[y_min, y_max]))

    else:
        fig.update_layout(width=200+(15*len(topics)),
                          height=height,
                          xaxis=dict(tickmode="array",
                                     ticktext=new_labels))
    return fig
