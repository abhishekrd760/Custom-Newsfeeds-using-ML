import pandas as pd
import numpy as np
import gspread
from sklearn.feature_extraction import text
from sklearn.svm import LinearSVC
import json
from oauth2client.service_account import ServiceAccountCredentials
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup


def get_text(x):
    soup = BeautifulSoup(x, 'lxml')
    text = soup.get_text()
    return text


df = pd.read_csv('newsdata.csv')
df.head()

my_words = []
my_stop_words = text.ENGLISH_STOP_WORDS.union(my_words)
print("removed Stop words by NLP are", my_stop_words)
vect = TfidfVectorizer(ngram_range=(1, 3), stop_words=set(my_stop_words), min_df=3)
tv = vect.fit_transform(df['text'])

clf = LinearSVC()
model = clf.fit(tv, df['wanted'])

scope = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/spreadsheets',
         "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
credentials = ServiceAccountCredentials.from_json_keyfile_name('CustomNewsfeed-00f5ae9c4fee.json', scope)
gc = gspread.authorize(credentials)

ws = gc.open("MyFeeds")
sh = ws.sheet1
zd = list(zip(sh.col_values(1), sh.col_values(2), sh.col_values(3), sh.col_values(4)))
zf = pd.DataFrame(zd, columns=['date', 'title', 'url', 'html'])
zf.replace('', pd.np.nan, inplace=True)
zf.dropna(inplace=True)
zf.loc[:,'text']=zf['html'].map(get_text)
zf.reset_index(drop=True, inplace=True)
test_matrix = vect.transform(zf['text'])

results = pd.DataFrame(model.predict(test_matrix),
                       columns=['wanted'])

rez = pd.merge(results, zf, left_index=True, right_index=True)

change_to_yes = [46, 50, 38, 21]
for i in rez.iloc[change_to_yes].index:
    rez.iloc[i]['wanted'] = 'y'

print("THE FOLLOWING ARE THE NEWS YOU WILL BE INTERESTED IN:  ")
count = 0
for i in range(len(rez)):
    if rez.loc[i, "wanted"] == 'y':
        count = count + 1
        print("news number: ", i)
        print("Title of the News: ", rez.loc[i, "title"])
        print(" ")
        print(" Brief About it: ", rez.loc[i, "text"])
        print(" ")
        print(" ")
print("Number Of News Given :", count)
