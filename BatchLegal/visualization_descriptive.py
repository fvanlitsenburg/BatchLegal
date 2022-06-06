import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import datetime
import pandas as pd
import numpy as np

def load_metadata_for_vis(filename):
    data = pd.read_csv(filename)
    data['date'] = pd.to_datetime(data['date'])
    data = data[~data["dir_1"].isna()].reset_index().drop(columns = "index") # drop rows that have NA in dir_1 column
    return data
    
def exploration_histogram(data):
    fig = plt.figure(figsize=(12,7))
    return sns.histplot(data['date'], bins = 50)

def exploration_valcounts(data, directory=1):
    '''
    directories with the number of publications
    '''
    return data[f'dir_{str(directory)}'].value_counts()

def exploration_list_subdirs(data, dir_1=None, dir_2=None):
    '''
    print possible subdir-keywords
    '''
    if dir_1 == None:
        print("Printing top-layer directories:")
        return list(data['dir_1'].value_counts().index)
    elif np.logical_and(dir_1 != None, dir_2 == None):
        print(f"subdirectories of dir_1: {dir_1}")
        return list(data[data['dir_1'] == dir_1]['dir_2'].value_counts().index)
    else: 
        print(f"subdirectories of dir_2: {dir_2}")
        return list(data[data['dir_2'] == dir_2]['dir_3'].value_counts().index)

def subset_data_subdirs(data, dir_1=None, dir_2=None, dir_3=None):
    '''
    used for visualizing the sub-directories - run "exploration_show_subdirs for the possible keywords"
    '''
    # create directory-index for the selected layer
    if dir_1 == None:
        print("no subsetting done, select keyword for dir_1 or 2")
        return data, 1
    elif np.logical_and(dir_1 != None, dir_2 == None):
        dir_keyword = dir_1
        dir_string = 'dir_1'
    elif np.logical_and(np.logical_and(dir_1 != None, dir_2 != None), dir_3 == None):
        dir_keyword = dir_2
        dir_string = 'dir_2'
    elif dir_3 != None:
        dir_keyword = dir_3
        dir_string = 'dir_3'
    dir_index = data[dir_string].value_counts().index
    return data[data[dir_string] == dir_keyword], int(dir_string[-1])+1

def subset_data(data, start_date="2011-01-01", end_date="2021-12-31", timesampling="Y", directory_level=1):
    '''
    filtering dir_1 data on start- and end-time and whether 1 datapoint per month or per year
    '''
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    data_subset = data[np.logical_and(data['date'] >= start_date, data['date'] <= end_date)]
    directory_code = f"dir_{directory_level}"
    # categories (dir_1) in descending frequencies
    dir_index = data[directory_code].value_counts().index
    #create dataframe for the others to append to and rename col to dir name
    try:   
        df = data_subset[data_subset[directory_code] == dir_index[0]].resample(timesampling, on='date')['title'].count().reset_index().rename(columns={'title':dir_index[0]})
    except IndexError:
        print("No Sub-Directories for this Keyword")
        return None
    # create dataframe with publications per directory
    for i in range(1,len(dir_index)):
        category = dir_index[i]
        temp = data_subset[data_subset[directory_code] == category].resample(timesampling, on='date')['title'].count().reset_index().rename(columns={'title':category})
        df = df.merge(temp, how='left', on='date').fillna(0)
    data_publications = pd.concat([df['date'], df.drop(columns = "date").astype('Int64')], axis=1)
    return data_publications

def visualization_piechart(data_publications):
    '''
    Comparison of Directory Frequency in Pie Chart
    '''
    piedata = data_publications.drop(columns='date').sum().reset_index()
    fig = px.pie(piedata, values=0, names='index', title='Directories of published documents')
    return fig#.show()

def visualization_stackedarea(data_publications, plottype="plotly"):
    '''
    stacked area plot in either plotly (interactive) or matplotlib
    '''
    # prepare data
    x = data_publications['date'].tolist() 
    y = data_publications.drop(columns = {"date"}).T.values.tolist()
    labels = data_publications.drop(columns = {"date"})
    # matplotlib
    if plottype == "matplotlib":
        fig = plt.figure(figsize=(12,7))
        plt.stackplot(x,y, labels=labels)
        plt.legend()
        plt.xlabel("Date of Publication")
        plt.ylabel("Number of Publications")
        plt.title(f"Publication of EU-Regulations per Directory (stacked)")
        return plt#.show()  
    # plotly
    elif plottype == "plotly":
        # create dict for the labels in plotly
        newnames = {}
        for index in range(0,len(labels.columns)):
            newnames[f"wide_variable_{str(index)}"] = labels.columns[index]
        # plot
        x_plot = x.copy()
        y_plot = y.copy()
        fig = px.area(x=x_plot, y=y_plot,
                      labels={"x": "Date of Publication",
                             "value": "Number of Publications",
                             "variable": "Category"},
                      title='Publication of EU-Regulations per Directory (stacked)')
        fig.for_each_trace(lambda t: t.update(name = newnames[t.name],
                                              legendgroup = newnames[t.name],
                                              hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name])))
        return fig#.show()
    else:
        return "please select either 'matplotlib' or 'plotly' as plottype"
    
def visualization_stackedarea_normalized(data_publications, plottype="plotly"):
    '''
    normalized stacked area plot in either plotly (interactive) or matplotlib
    '''
    #normalize 
    df = data_publications.drop(columns = {'date'})
    data_publications_normalized = df.div(df.sum(axis=1), axis=0)
    y_norm = data_publications_normalized.T.values.tolist()
    # prepare data
    x_norm = data_publications['date'].tolist() 
    labels = data_publications.drop(columns = {"date"})
    # matplotlib
    if plottype == "matplotlib":
        # matplotlib
        fig = plt.figure(figsize=(12,7))
        plt.stackplot(x_norm, y_norm, labels=labels)
        plt.legend()
        plt.xlabel("Date of Publication")
        plt.ylabel("Share of Publications in this Directory")
        plt.title(f"Publication of EU-Regulations per Directory (stacked and normalized)")
        return plt#.show()
    # plotly
    elif plottype == "plotly":
        # create dict for the labels in plotly
        newnames = {}
        for index in range(0,len(labels.columns)):
            newnames[f"wide_variable_{str(index)}"] = labels.columns[index]
        # plot
        fig = px.area(x=x_norm, y=y_norm,
              labels={"x": "Date of Publication",
                     "value": "Share of Publications in this Directory",
                     "variable": "Category"},
              title='Publication of EU-Regulations per Directory (stacked and normalized)')
        fig.for_each_trace(lambda t: t.update(name = newnames[t.name],
                                              legendgroup = newnames[t.name],
                                              hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name])))
        return fig#.show()
    else:
        return "please select either 'matplotlib' or 'plotly' as plottype"
    
    
if __name__ == "__main__":
    filename = "../raw_data/20220602.csv"
    data = load_metadata_for_vis(filename)
    data, dirlevel = subset_data_subdirs(data, dir_1=None, dir_2=None, dir_3=None)
    data = subset_data(data, start_date="2011-01-01", end_date="2021-12-31", timesampling="Y", directory_level=dirlevel)
#    visualization_piechart(data)