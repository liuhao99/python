# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 10:21:14 2020

@author: a0866568
"""

import requests
import lxml.html as lh
import pandas as pd

#from requests.adapters import HTTPAdapter
#from requests.packages.urllib3.util.retry import Retry


#session = requests.Session()
#retry = Retry(connect=3, backoff_factor=0.5)
#adapter = HTTPAdapter(max_retries=retry)
#session.mount('http://', adapter)
#session.mount('https://', adapter)

#session.get(url)

url = 'https://tpwd.texas.gov/fishboat/fish/management/stocking/trout_stocking.phtml'
#Create a handle, page, to handle the contents of the website

#url_p = 'http://pokemondb.net/pokedex/all'
#page = session.get(url)
#
##Store the contents of the website under doc
#doc = lh.fromstring(page.content)
##Parse data that are stored between <tr>..</tr> of HTML
#tr_elements = doc.xpath('//tr')

try:
    r = requests.get(url,timeout=3)
    r.raise_for_status()
except requests.exceptions.HTTPError as errh:
    print ("Http Error:",errh)
except requests.exceptions.ConnectionError as errc:
    print ("Error Connecting:",errc)
except requests.exceptions.Timeout as errt:
    print ("Timeout Error:",errt)
except requests.exceptions.RequestException as err:
    print ("OOps: Something Else",err)

#Store the contents of the website under doc
doc = lh.fromstring(r.content)
#Parse data that are stored between <tr>..</tr> of HTML
tr_elements = doc.xpath('//tr')
    
    
#df_list = pd.read_html(r)
#df = df_list[-1]
#print(df)
#df.to_csv('trout.csv')

#Check the length of the first 12 rows
[len(T) for T in tr_elements[:12]]
tr_elements = doc.xpath('//tr')
#Create empty list
col=[]
i=0
#For each row, store each first element (header) and an empty list
for t in tr_elements[0]:
    i+=1
    name=t.text_content()
    col.append((name,[]))
    
    
#Since out first row is the header, data is stored on the second row onwards
for j in range(1,len(tr_elements)):
    #T is our j'th row
    T=tr_elements[j]
    
    #If row is not of size 10, the //tr data is not from our table 
    if len(T)!=5:
        break
    
    #i is the index of our column
    i=0
    
    #Iterate through each element of the row
    for t in T.iterchildren():
        data=t.text_content() 
        #Check if row is empty
        if i>0:
        #Convert any numerical value to integers
            try:
                data=int(data)
            except:
                pass
        #Append the data to the empty list of the i'th column
        col[i][1].append(data)
        #Increment i for the next column
        i+=1    

[len(C) for (title,C) in col]
    
Dict={title:column for (title,column) in col}
df=pd.DataFrame(Dict)    