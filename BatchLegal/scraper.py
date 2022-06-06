import pandas as pd
import numpy as np
import requests
import datetime
from bs4 import BeautifulSoup

'''
This python code is meant to read metadata of EU-Regulations (especially their cellar-number) and get the content of that Regulation. 
The input is taken from the api.py script (dataframe of metadata including date, title, cellar-number...)
The output is the same metadata-csv with one more column: the content of the Regulation.
'''

def read_metadata(filename):
    '''
    reads in metadata from the api.py file and adds an empty column where the content of the pages will be in the end
    '''
    data = pd.read_csv(filename)
    data['Content'] = None
    return data

def get_url(cellar_ref, doctype="03"):
    '''
    creates a url based on the cellar reference in the metadata, which will be used to scrape the content
    '''
    psid = cellar_ref
    psname = "cellar" # other options: cellar, celex, oj...
    lancode = "0006" # language code
    doctype = doctype # default: 03
    docnum = "DOC_1"
    # for further information, see Documentation Page 37: https://op.europa.eu/en/publication-detail/-/publication/50ecce27-857e-11e8-ac6a-01aa75ed71a1/language-en/format-PDF/source-73059305
    return f"http://publications.europa.eu/resource/{psname}/{psid}.{lancode}.{doctype}/{docnum}"

def get_content(URL):
    '''
    main function, scrapes content. added some code to catch errors.
    '''
    response = requests.get(URL, headers={"Accept-Language":"en-US"})
    # one minor bug still in there: some requests (for example number 58 in 20220601_larger_data_b) are a valid request but have to download many mb first. the solution would be to stop the request.get if it runs longer than x seconds
    try:
        soup = BeautifulSoup(response.content, "html.parser")
        if str(soup)[1:4] == "PDF":
            '''
            in some (few) cases, the doctype is not 03 but 02. change it for these cases
            '''
            print("pdf detected, but fixed")
            doctype = '02'
            URL = URL[:-8] + doctype + URL[-6:]
            response = requests.get(URL, headers={"Accept-Language":"en-US"})
            soup = BeautifulSoup(response.content, "html.parser")
        else:
            print("no problem here")
            doctype = '03'
    except:
        '''
        in case there is an error
        '''
        print("yes problem here")
        URL = URL[:-8] + '02' + URL[-6:]
        response = requests.get(URL, headers={"Accept-Language":"en-US"})
        soup = BeautifulSoup(response.content, "html.parser")
        
    if soup.find("p", class_="oj-normal") == None:
        content = ' '.join([item.text for item in soup.find_all("p", class_="normal")])
    else:
        content = ' '.join([item.text for item in soup.find_all("p", class_="oj-normal")])
    return content

def clean_data(data):
    '''
    takes scraped data and removes rows which contain, no information, information in non-english and the head of all the valid content
    '''
    data = data[data['Content'] != ""]
    data = data[data['Content'].str[0:3] == 'THE'] #remove content in other languages
    data = data[data['Content'].str.contains('Whereas: ')] # contains the split word
    data.loc[:, 'Content'] = data['Content'].apply(lambda x: x.split('Whereas: ', 1)[1]) # split off header
    data = data[data['Content'].str[0:3] == "(1)"] #gotta make sure it's standardized!
    return data.reset_index().drop(columns = "index")

def get_all_content(data):
    '''
    loops over the functions to get all content
    '''
    cellar_references = data['cellar']    
    for index, ref in enumerate(cellar_references):
        data.loc[index, 'Content'] = get_content(get_url(ref))
        print(f'Row {index} with cellar-number {ref} done')
    return data

if __name__ == '__main__':
    # try reading metadata file, if non-existant request it from api.py
    path = "../raw_data/"
    filename_without_csv = "20220602"
    try:
        data = read_metadata(path + filename_without_csv + '.csv')
    except:
        from BatchLegal.api import generate_meta
        data,requester = generate_meta(start_year=2010,end_year=2022)  
    #data = data.iloc[:20]
    data_with_content = get_all_content(data)
    data_with_content_clean = clean_data(data_with_content)
    data_with_content_clean.to_csv(path + filename_without_csv + '_scraped_test.csv')
