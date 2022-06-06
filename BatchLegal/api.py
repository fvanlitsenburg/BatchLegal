import pandas as pd
import numpy as np
import requests
import math
import time
from xml.etree import ElementTree as et
from dotenv import load_dotenv, find_dotenv
import os

env_path = find_dotenv()

load_dotenv(env_path)

URL = 'https://eur-lex.europa.eu/EURLexWebService'

username = os.getenv('API_USERNAME')

passwrd = os.getenv('API_PASSWORD')

'''
This Python script queries the EURLexWebservice. To use it yourself, you need to request a username and password. This can be done here:
https://eur-lex.europa.eu/content/help/data-reuse/webservice.html

This package uses a two-step process to obtain the data from the EUR-Lex webservice. First, we use the EURLex Web Service to obtain MetaData and *Cellar references*.

Cellar is the repository of the content for documents. More information: https://eur-lex.europa.eu/content/help/data-reuse/reuse-contents-eurlex-details.html

In the package "scraping" we then use the Cellar references to attach the text content to each document

'''

def get_pages(request_in):
    '''
    Gets the number of pages required to complete a request to the EURLex Web Service.

    Parameters:
    A request from the requests package to the EUR-Lex WebService, so that the number of pages and hits per page can be calculated.
    '''
    newroot = et.fromstring(request_in.content)
    hits = newroot.find(".//{http://eur-lex.europa.eu/search}totalhits").text
    hitsppage = newroot.find(".//{http://eur-lex.europa.eu/search}numhits").text
    print(hits)
    print(hitsppage)
    return math.ceil(int(hits)/int(hitsppage))

def request_body(page_num,start_year,end_year):
    '''
    Returns the body for a EURLex Web Service request to get all 'Regulation' documents *published* between start_year and end_year, for the page_num page.

    Parameters:
    page_num: page of the request
    start_year: start year of the request
    end_year: end year of the request
    '''
    strt = str(start_year)
    stp = str(end_year)
    body = '''<soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope" xmlns:sear="http://eur-lex.europa.eu/search">
      <soap:Header>
        <wsse:Security xmlns:wsse="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd" soap:mustUnderstand="true">
          <wsse:UsernameToken xmlns:wsu="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd" wsu:Id="UsernameToken-1">
            <wsse:Username>'''+username+'''</wsse:Username>
            <wsse:Password Type="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordText">'''+passwrd+'''</wsse:Password>
          </wsse:UsernameToken>
        </wsse:Security>
      </soap:Header>
      <soap:Body>
        <sear:searchRequest>
          <sear:expertQuery><![CDATA[DTS_SUBDOM = LEGISLATION AND (FM_CODED = REG OR REG_DEL OR REG_FINANC OR REG_IMPL) AND DTS_SUBDOM = LEGISLATION AND (PD >= 01/01/'''+strt+'''  <= 31/12/'''+stp+''' OR PD >= 01/01/'''+strt+'''  <= 31/12/'''+stp+''')]]></sear:expertQuery>
          <sear:page>'''+str(page_num)+'''</sear:page>
          <sear:pageSize>20</sear:pageSize>
          <sear:searchLanguage>en</sear:searchLanguage>
        </sear:searchRequest>
      </soap:Body>
    </soap:Envelope>

    '''
    return body

def full_request(start_year,end_year):
    ''' Returns a list of requests. Each request is a different page of the query to fetch all Regulations published between start_year and end_year on the EUR-Lex Webservice.

    Parameters:
    start_year: start year of the request
    end_year: end year of the request

    '''
    reqs = []
    headerdict = {'Content-Type':'application/soap+xml',
              'Content-Length':'0',
              'Accept':'*/*',
              'Connection':'keep-alive'
             }
    temp = requests.post(URL,headers=headerdict,data=request_body(1,start_year,end_year))
    reqs.append(temp)
    pages = get_pages(temp)
    print(str(pages) + " pages to crawl")
    for i in range(2,pages):
        temp = requests.post(URL,headers=headerdict,data=request_body(i,start_year,end_year))
        reqs.append(temp)
        time.sleep(3)
    return(reqs)

def parse_xml(request):

    ''' Takes an XML returned from request and turns it into a dataframe with:
    - Cellar ID
    - Date
    - Dir code
    - Name of the dirs

    Input:
    A request

    '''

    root = et.fromstring(request.content)

    if root[0].tag != '{http://www.w3.org/2003/05/soap-envelope}Body':

        pd_dict = {'title':[],'cellar':[],'date':[],'dir_code':[],'dir_1':[],'dir_2':[],'dir_3':[],'dir_4':[],'dir_5':[],'dir_6':[]}

        for child in root[1][0].findall('{http://eur-lex.europa.eu/search}result'):
                # Get reference
                raw_ref = child.find("./{http://eur-lex.europa.eu/search}reference").text
                ref = str.replace(raw_ref,"eng_cellar:","")
                pd_dict['cellar'].append(ref[0:-3])

                # Get date

                pd_dict['date'].append(child.find(".//{http://eur-lex.europa.eu/search}DATE_PUBLICATION")[0].text)

                # Get dir_code and dir_names

                dirs = child.find(".//{http://eur-lex.europa.eu/search}RESOURCE_LEGAL_IS_ABOUT_CONCEPT_DIRECTORY-CODE")
                if dirs == None:
                    pd_dict['dir_code'].append("")
                    for i in range (1,6+1):
                        entry = 'dir_'+str(i)
                        pd_dict[entry].append("")
                else:
                    pd_dict['dir_code'].append(dirs[-1][0].text)
                    for i in range(0,len(dirs)):
                        entry = 'dir_'+str(i+1)
                        pd_dict[entry].append(dirs[i][2].text)
                    for i in range (len(dirs)+1,6+1):
                        entry = 'dir_'+str(i)
                        pd_dict[entry].append("")

                # Get title
                title = child.find(".//{http://eur-lex.europa.eu/search}EXPRESSION_TITLE")
                if title == None:
                    pd_dict['title'].append("")
                else:
                    if len(title) == 1:
                        pd_dict['title'].append(title[0].text)
                    else:
                        pd_dict['title'].append(title[1].text)

        return pd_dict
    else:
        pass

def generate_meta(start_year=2010,end_year=2022):
    '''
    Returns a dataframe of metadata, as well as a list of requests.

    Parameters:
    start_year: start year of the request
    end_year: end year of the request
    '''
    requester = full_request(start_year,end_year)
    df = pd.DataFrame(parse_xml(requester[0]))
    for i in requester[1:]:
        tempdf = pd.DataFrame(parse_xml(i))
        df = pd.concat([df,tempdf])
    return df,requester
