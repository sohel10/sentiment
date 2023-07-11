#!/usr/bin/env python
# coding: utf-8

# # Text Classification for Sentiment and Emotion Analysis: A Comprehensive Guide to NLP with Machine Learning and Deep Learning, Including Web Model Deployment

# ![Natural%20Language%20Processing%28NLP%29.png](attachment:Natural%20Language%20Processing%28NLP%29.png)

# In[1]:


import pandas as pd


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


import warnings
warnings.filterwarnings('ignore')


# In[254]:


data = pd.read_csv('sentiments.csv')


# In[255]:


data


# In[256]:


# Drop the unnecesary column from dataset
data = data.drop(['Unnamed: 0', 'Tweet Id'], axis=1)


# In[257]:


data


# In[258]:


data.shape


# In[259]:


data.columns


# In[260]:


data.duplicated().sum()


# In[261]:


data = data.drop_duplicates()


# In[262]:


sns.heatmap(data.corr(),annot=True,cmap='RdYlGn',linewidths=0.2,annot_kws={'size':20})
fig=plt.gcf()
fig.set_size_inches(6,4)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()


# In[263]:


def summary(data):
    print(f"Dataset has {data.shape[1]} features and {data.shape[0]} examples.")
    summary = pd.DataFrame(index=data.columns)
    summary["Unique"] = data.nunique().values
    summary["Missing"] = data.isnull().sum().values
    summary["Duplicated"] = data.duplicated().sum()
    summary["Types"] = data.dtypes
    return summary


# In[264]:


summary(data)


# In[265]:


custom_colors = [
    (100/255, 108/255, 116/255),   # nevada
    (228/255, 12/255, 33/255),     # red-ribbon
    (68/255, 68/255, 76/255),      # abbey
    (172/255, 28/255, 44/255),     # roof-terracotta 
]
custom_palette = sns.color_palette(custom_colors)

plt.figure(figsize=(6,6))
data1 = data['County_Positive'].value_counts().values
labels = ['Negative', 'Positive']
plt.pie(data1, labels = labels, colors = custom_palette, autopct='%.0f%%')
plt.show()
# In[266]:


# Let's write a functin to print the total percentage of the missing values.(this can be a good exercise for beginners to try to write simple functions like this.)
def missing_percentage(data):
    """This function takes a DataFrame(df) as input and returns two columns, total missing values and total missing values percentage"""
    total = data.isnull().sum().sort_values(ascending = False)
    percent = round(data.isnull().sum().sort_values(ascending = False)/len(data)*100,2)
    return pd.concat([total, percent], axis=1, keys=['Total','Percent'])


# In[267]:


missing_percentage(data)


# In[268]:


def percent_value_counts(data, feature):
    """This function takes in a dataframe and a column and finds the percentage of the value_counts"""
    percent = pd.DataFrame(round(data.loc[:,feature].value_counts(dropna=False, normalize=True)*100,2))
    ## creating a df with th
    total = pd.DataFrame(data.loc[:,feature].value_counts(dropna=False))
    ## concating percent and total dataframe

    total.columns = ["Total"]
    percent.columns = ['Percent']
    return pd.concat([total, percent], axis = 1)


# In[269]:


percent_value_counts(data, 'sentiment')


# In[270]:


percent_value_counts(data, 'emotion')


# In[ ]:





# # Data Visualization

# In[271]:


data['sentiment'].unique()


# In[272]:


data['sentiment'].value_counts()


# In[273]:


import plotly.express as px

fig = px.histogram(data, x='sentiment', color='sentiment', title='Sentiment Distribution')
fig.update_layout(showlegend=False)
fig.show()


# In[274]:


label_data = data['sentiment'].value_counts()

explode = (0.1, 0.1, 0.1)
plt.figure(figsize=(14, 10))
patches, texts, pcts = plt.pie(label_data,
                               labels = label_data.index,
                               colors = ['blue', 'red', 'green'],
                               pctdistance = 0.65,
                               shadow = True,
                               startangle = 90,
                               explode = explode,
                               autopct = '%1.1f%%',
                               textprops={ 'fontsize': 25,
                                           'color': 'black',
                                           'weight': 'bold',
                                           'family': 'serif' })
plt.setp(pcts, color='white')

hfont = {'fontname':'serif', 'weight': 'bold'}
plt.title('Label', size=20, **hfont)

centre_circle = plt.Circle((0,0),0.40,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.show()


# In[275]:


fig = px.histogram(data, x='emotion', color='emotion', title='Emotion Distribution')
fig.update_layout(showlegend=False)
fig.show()


# In[276]:


label_data = data['emotion'].value_counts()

explode = (0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.4, 0.5)
plt.figure(figsize=(14, 10))
patches, texts, pcts = plt.pie(label_data,
                               labels = label_data.index,
                               colors = ['blue', 'red', 'green', 'orange', 'pink', 'yellow', 'violet', 'grey'],
                               pctdistance = 0.65,
                               shadow = True,
                               startangle = 0,
                               explode = explode,
                               autopct = '%1.1f%%',
                               textprops={ 'fontsize': 15,
                                           'color': 'black',
                                           'weight': 'bold',
                                           'family': 'serif' })
plt.setp(pcts, color='black')

hfont = {'fontname':'serif', 'weight': 'bold'}
plt.title('Label', size=20, **hfont)

centre_circle = plt.Circle((0,0),0.40,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.show()


# In[277]:


fig = px.histogram(data, x='emotion', color='emotion', title='Emotion Distribution')
fig.update_layout(showlegend=False)
fig.show()


# In[278]:


fig = px.histogram(data, x='emotion', color='emotion')
fig.update_layout(title='Emotion Distribution', showlegend=False)
fig.show()


# In[ ]:


fig = px.histogram(data, x='emotion', color='emotion', title='Emotion Distribution',
                   category_orders={'emotion': data['emotion'].value_counts().index})
fig.update_layout(showlegend=False)
fig.show()


# In[ ]:


import plotly.express as px
fig = px.pie(data, names='emotion', title='Emotion Distribution')
fig.update_layout(showlegend=False)
fig.show()


# In[ ]:


fig = px.histogram(data, x='sentiment_score', nbins=20, marginal='rug', histnorm='density', title='Sentiment Score Distribution')
fig.update_layout(showlegend=False)
fig.show()



# In[ ]:


import plotly.express as px

fig = px.histogram(data, x='sentiment_score', nbins=20, histnorm='density', 
                   title='Sentiment Score Distribution', marginal='box')
fig.update_layout(showlegend=False)
fig.show()



# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 6))
sns.distplot(data['sentiment_score'], bins=20, kde=True)
plt.xticks(rotation=0)
plt.title('Sentiment Score Distribution')
plt.show()


# In[284]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 6))
sns.histplot(data['sentiment_score'], bins=20, kde=True)
plt.xticks(rotation=0)
plt.title('Sentiment Score Distribution')
plt.show()


# In[285]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 6))
sns.distplot(data['sentiment_score'], bins=20, kde=True)
plt.xticks(rotation=0)
plt.title('Sentiment Score Distribution')
plt.show()



# In[286]:


fig = px.histogram(data_frame=data, x='sentiment_score', nbins=20, color_discrete_sequence=['lightskyblue'])
fig.update_layout(
    title="Sentiment Score Histogram",
    xaxis_title="Sentiment Score",
    yaxis_title="Frequency",
    xaxis=dict(tickangle=0),
    bargap=0.2,
    bargroupgap=0.1,
    autosize=False,
    width=800,
    height=400
)

fig.show()


# In[287]:


plt.figure(figsize=(15,6))
sns.boxplot(data['sentiment_score'], data = data, palette = 'hls')
plt.xticks(rotation = 0)
plt.show()


# In[ ]:


import plotly.express as px

fig = px.box(data, y='sentiment_score')
fig.update_layout(title='Box Plot of Sentiment Scores')
fig.show()


# In[ ]:


fig = px.box(data, y='emotion_score')
fig.update_layout(title='Box Plot of Emotion Scores')
fig.show()


# In[ ]:


fig = px.violin(data, y='sentiment_score')
fig.update_layout(title='Violin Plot of Sentiment Scores')
fig.show()


# In[ ]:


fig = px.violin(data, y='emotion_score')
fig.update_layout(title='Violin Plot of Emotion Scores')
fig.show()


# In[288]:


plt.figure(figsize=(15,6))
sns.barplot(x = data['sentiment'], y = data['sentiment_score'], data = data, ci = None, palette = 'hls')
plt.show()


# In[289]:


import plotly.express as px
import plotly.colors

fig = px.bar(
    data_frame=data,
    x='sentiment',
    y='sentiment_score',
    color='sentiment',
    color_continuous_scale=plotly.colors.sequential.deep,
)

fig.update_layout(
    title="Sentiment Scores by Sentiment",
    xaxis_title="Sentiment",
    yaxis_title="Sentiment Score",
    autosize=False,
    width=800,
    height=500
)

fig.show()


# In[290]:


plt.figure(figsize=(15,6))
sns.barplot(x = data['emotion'], y = data['emotion_score'], data = data, ci = None, palette = 'hls')
plt.show()


# In[291]:


plt.figure(figsize=(15,6))
sns.boxplot(x = data['emotion'], y = data['emotion_score'], data = data, palette = 'hls')
plt.show()


# In[292]:


data['Datetime'] = pd.to_datetime(data['Datetime'])


# In[293]:


data


# In[294]:


data.info()


# In[295]:


data.columns


# # Feature Engineering

# In[296]:


data['Datetime'] = pd.to_datetime(data['Datetime'])


# In[297]:


import pandas as pd

data['Datetime'] = pd.to_datetime(data['Datetime'])

data['Day'] = data['Datetime'].dt.day
data['Month'] = data['Datetime'].dt.month
data['Year'] = data['Datetime'].dt.year


# In[298]:


data


# In[299]:


# Group by day and calculate average sentiment score
daily_sentiment = data.groupby(data['Day'])['sentiment_score'].mean()

# Visualize daily sentiment scores
plt.figure(figsize=(10, 6))
plt.plot(daily_sentiment.index, daily_sentiment.values)
plt.xlabel('Date')
plt.ylabel('Average Sentiment Score')
plt.title('Daily Sentiment Analysis')
plt.show()


# In[300]:


daily_emotion = data.groupby(data['Day'])['emotion_score'].mean()
plt.figure(figsize=(10, 6))
plt.plot(daily_emotion.index, daily_emotion.values)
plt.xlabel('Date')
plt.ylabel('Average Emotion Score')
plt.title('Daily Emotion Analysis')
plt.show()


# In[301]:


# Group by day and calculate average sentiment score
daily_sentiment = data.groupby(data['Day'])['sentiment_score'].mean().reset_index()

# Visualize daily sentiment scores
fig = px.line(daily_sentiment, x='Day', y='sentiment_score', title='Daily Sentiment Analysis')
fig.update_layout(xaxis_title='Date', yaxis_title='Average Sentiment Score')
fig.show()


# In[302]:


daily_sentiment


# In[303]:


# Group by day and calculate average emotion score
daily_emotion = data.groupby(data['Day'])['emotion_score'].mean().reset_index()

# Visualize daily emotion scores
fig = px.line(daily_emotion, x='Day', y='emotion_score', title='Daily Emotion Analysis')
fig.update_layout(xaxis_title='Date', yaxis_title='Average Emotion Score')
fig.show()


# In[304]:


data1=data.copy()


# In[305]:


# Prepare the data
data1.set_index('Datetime', inplace=True)

# Calculate daily average sentiment score
daily_sentiment = data1['sentiment_score'].resample('D').mean()

# Plot daily sentiment analysis
fig = px.line(daily_sentiment, x=daily_sentiment.index, y=daily_sentiment.values)
fig.update_layout(
    title="Daily Sentiment Analysis",
    xaxis_title="Date",
    yaxis_title="Average Sentiment Score"
)
fig.show()

# Calculate daily average emotion score
daily_emotion = data1['emotion_score'].resample('D').mean()

# Plot daily emotion analysis
fig = px.line(daily_emotion, x=daily_emotion.index, y=daily_emotion.values)
fig.update_layout(
    title="Daily Emotion Analysis",
    xaxis_title="Date",
    yaxis_title="Average Emotion Score"
)
fig.show()


# In[306]:


data2=data.copy()


# In[307]:


# Prepare the data
data2.set_index('Datetime', inplace=True)

# Resample the data to get daily sentiment scores for each category
daily_sentiments = data2.groupby(['Day', 'sentiment'])['sentiment_score'].mean().unstack()

# Plot the comparative analysis
fig = go.Figure()

# Add traces for each sentiment category
for sentiment in ['positive', 'negative', 'neutral']:
    fig.add_trace(go.Scatter(
        x=daily_sentiments.index,
        y=daily_sentiments[sentiment],
        mode='lines',
        name=sentiment.capitalize()
    ))

fig.update_layout(
    title="Comparative Sentiment Analysis",
    xaxis_title="Date",
    yaxis_title="Average Sentiment Score",
    legend_title="Sentiment"
)
fig.show()


# In[308]:


data3=data.copy()


# In[309]:


# Prepare the data
data3.set_index('Datetime', inplace=True)

# Resample the data to get daily emotion scores for each category
daily_emotions = data3.groupby(['Day', 'emotion'])['emotion_score'].mean().unstack()

# Plot the comparative analysis
fig = go.Figure()

# Add traces for each emotion category
for emotion in ['anticipation', 'joy', 'anger', 'sadness', 'fear', 'optimism',
       'disgust', 'surprise']:
    fig.add_trace(go.Scatter(
        x=daily_emotions.index,
        y=daily_emotions[emotion],
        mode='lines',
        name=emotion.capitalize()
    ))

fig.update_layout(
    title="Comparative Emotion Analysis",
    xaxis_title="Date",
    yaxis_title="Average Emotion Score",
    legend_title="Emotion"
)
fig.show()


# In[310]:


data


# In[311]:


data4=data.copy()


# In[312]:


data4.columns


# In[313]:


data4 = data[['Text', 'sentiment', 'sentiment_score',
       'emotion', 'emotion_score']]


# In[314]:


data4


# # Data Preprocessing / Data Cleaning

# In[315]:


def clean_text(text):
    text = text.lower() 
    return text.strip()


# In[316]:


data4.Text = data4.Text.apply(lambda x: clean_text(x))


# In[317]:


data4['Text']


# In[318]:


import string
string.punctuation


# In[319]:


def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree


# In[320]:


data4.Text= data4['Text'].apply(lambda x:remove_punctuation(x))


# In[321]:


data4['Text']


# In[322]:


import re
def tokenization(text):
    tokens = re.split('W+',text)
    return tokens


# In[323]:


data['Text']= data4['Text'].apply(lambda x: tokenization(x))


# In[324]:


data4['Text']


# In[325]:


import nltk
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')


# In[326]:


stopwords


# In[327]:


def remove_stopwords(text):
    output= " ".join(i for i in text if i not in stopwords)
    return output


# In[328]:


data4['Text']= data4['Text'].apply(lambda x:remove_stopwords(x))


# In[329]:


data4['Text']


# In[330]:


from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()


# In[331]:


nltk.download('wordnet')


# In[332]:


nltk.download('omw-1.4')


# In[333]:


def lemmatizer(text):
    lemm_text = "".join([wordnet_lemmatizer.lemmatize(word) for word in text])
    return lemm_text


# In[334]:


data4['Text']=data4['Text'].apply(lambda x:lemmatizer(x))


# In[335]:


data4['Text']


# In[336]:


def clean_text(text):
    text = re.sub('\[.*\]','', text).strip() 
    text = re.sub('\S*\d\S*\s*','', text).strip()  
    return text.strip()


# In[337]:


data4['Text'] = data4.Text.apply(lambda x: clean_text(x))


# In[338]:


data4['Text']


# In[339]:


import spacy
nlp = spacy.load('en_core_web_sm')


# In[340]:


stopwords = nlp.Defaults.stop_words
def lemmatizer(text):
    doc = nlp(text)
    sent = [token.lemma_ for token in doc if not token.text in set(stopwords)]
    return ' '.join(sent)


# In[341]:


data4['Text'] =  data4.Text.apply(lambda x: lemmatizer(x))


# In[342]:


def remove_urls(vTEXT):
    vTEXT = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', vTEXT, flags=re.MULTILINE)
    return(vTEXT)


# In[343]:


data4['Text'] = data4.Text.apply(lambda x: remove_urls(x))


# In[344]:


data4['Text']


# In[345]:


def remove_digits(text):
    clean_text = re.sub(r"\b[0-9]+\b\s*", "", text)
    return(text)


# In[346]:


data4['Text'] = data4.Text.apply(lambda x: remove_digits(x))


# In[347]:


data4['Text']


# In[348]:


def remove_digits1(sample_text):
    clean_text = " ".join([w for w in sample_text.split() if not w.isdigit()]) 
    return(clean_text)


# In[349]:


data4['Text'] = data4.Text.apply(lambda x: remove_digits1(x))


# In[350]:


data4['Text']


# In[351]:


def remove_emojis(data):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    return re.sub(emoji_pattern, '', data)


# In[352]:


import re

def remove_emojis(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


# In[353]:


data4['Text'] = data['Text'].astype(str).apply(lambda x: remove_emojis(x))


# In[354]:


data4


# In[355]:


data4['Text_Length'] = data4['Text'].apply(lambda x: len(x))


# In[356]:


data4


# In[357]:


plt.figure(figsize=(15,6))
sns.histplot(data4['Text_Length'], kde = True, bins = 20, palette = 'hls')
plt.xticks(rotation = 0)
plt.show()


# In[358]:


plt.figure(figsize=(15,6))
sns.distplot(data4['Text_Length'], kde = True, bins = 20)
plt.xticks(rotation = 0)
plt.show()


# In[359]:


import wordcloud


# In[360]:


from wordcloud import WordCloud
data = data4['Text']
plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 1000 , width = 1600 , height = 800,
               collocations=False).generate(" ".join(data))
plt.imshow(wc)
plt.axis('off')
plt.show()


# In[361]:


data = data4[data4['sentiment']=="positive"]['Text']
plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 1000 , width = 1600 , height = 800,
               collocations=False).generate(" ".join(data))
plt.imshow(wc)
plt.axis('off')
plt.show()


# In[362]:


data = data4[data4['sentiment']=="negative"]['Text']
plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 1000 , width = 1600 , height = 800,
               collocations=False).generate(" ".join(data))
plt.imshow(wc)
plt.axis('off')
plt.show()


# In[363]:


data = data4[data4['sentiment']=="neutral"]['Text']
plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 1000 , width = 1600 , height = 800,
               collocations=False).generate(" ".join(data))
plt.imshow(wc)
plt.axis('off')
plt.show()


# In[364]:


data5 = data4[['Text','sentiment']]


# In[365]:


data5


# In[366]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# In[367]:


# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier


# In[368]:


X = data5['Text']
y = data5['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[369]:


vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


# In[370]:


naive_bayes = MultinomialNB()
naive_bayes.fit(X_train_vectorized, y_train)


# In[371]:


y_pred = naive_bayes.predict(X_test_vectorized)


# In[372]:


accuracy = accuracy_score(y_test, y_pred)
cr = classification_report(y_test, y_pred)


# In[373]:


print("Accuracy:", accuracy)
print("Classification Report:\n", cr)


# In[375]:


param_grid = {
    'alpha': [0.1, 0.5, 1.0],  # Smoothing parameter
    'fit_prior': [True, False]  # Whether to learn class prior probabilities or not
}


# In[376]:


from sklearn.model_selection import GridSearchCV


# In[377]:


grid_search = GridSearchCV(estimator=naive_bayes, param_grid=param_grid, cv=5)
grid_search.fit(X_train_vectorized, y_train)


# In[378]:


best_params = grid_search.best_params_
best_model = grid_search.best_estimator_


# In[379]:


y_pred = best_model.predict(X_test_vectorized)


# In[380]:


accuracy = accuracy_score(y_test, y_pred)


# In[381]:


accuracy


# In[382]:


cr = classification_report(y_test, y_pred)


# In[383]:


print(cr)


# In[384]:


data6 = data4[['Text','sentiment']]


# In[385]:


data6


# In[386]:


import xgboost as xgb


# In[387]:


from sklearn.preprocessing import LabelEncoder


# In[388]:


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(data6['sentiment'])


# In[389]:


X = data6['Text']
y = y_encoded


# In[390]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[391]:


vectorizer1 = TfidfVectorizer()
X_train_vectorized = vectorizer1.fit_transform(X_train)
X_test_vectorized = vectorizer1.transform(X_test)


# In[392]:


xg_model = xgb.XGBClassifier()
xg_model.fit(X_train_vectorized, y_train)


# In[393]:


y_pred = xg_model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report)


# In[394]:


# Random Forest


# In[395]:


model = RandomForestClassifier()


# In[396]:


model = RandomForestClassifier()
model.fit(X_train_vectorized, y_train)


# In[397]:


y_pred = model.predict(X_test_vectorized)


# In[398]:


from sklearn.metrics import classification_report

y_pred = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:")
print(report)


# In[399]:


import pickle

with open('naive_bayes_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)


# In[400]:


with open('tfidf_vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




