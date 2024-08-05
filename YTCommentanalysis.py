from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pandas as pd
import os
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()
API_key='Enter Your API key here'
youtube = build('youtube', 'v3', developerKey=API_key)
def get_comments(youtube,video_id):
    comments=[]
    try:
        results=youtube.commentThreads().list(
            part='snippet',videoId=video_id,textFormat='plainText'
        ).execute()
        while results:
            for item in results['items']:
                comment=item['snippet']['topLevelComment']['snippet']
                comments.append({
                    'Author_name': comment['authorDisplayName'],
                    'Text': comment['textDisplay'],
                    'Published_at': comment['publishedAt'],
                    'Like_count': comment['likeCount']})
            if 'nextPageToken' in results:
                results=youtube.commentThreads().list(
                    part='snippet',videoId=video_id,textFormat='plainText',pageToken= results['nextPageToken']
                ).execute()
            else:
                break
    except HttpError as e:
        print(f"An HTTP error {e.resp.status} occurred: {e.content}")
    return comments
def Sentiment_analysis(comments):
    sentiments = {'positive': 0, 'negative': 0, 'neutral': 0}
    for comment in comments:
        text=comment['Text']
        scores = sid.polarity_scores(text)
        if scores['compound'] >= 0.05:
            sentiments['positive']+=1
            comment['Polarity']='Positive'
        elif scores['compound'] <= -0.05:
            sentiments['negative']+=1
            comment['Polarity']='Negative'
        else:
            sentiments['neutral']+=1
            comment['Polarity']='Neutral'
    return sentiments
def plotSentiments(sentiments):
    labels = sentiments.keys()
    sizes = sentiments.values()
    colors = ['#41DE2F','#D7401B','#93D4E9']
    explode = (0.1, 0, 0)
    plt.figure(figsize=(6,6))
    plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%',shadow=True)
    plt.axis('equal')
    plt.title('Sentiment Analysis of YouTube Comments')
    plt.show()
if __name__=='__main__':
    video_url=input("Enter The Video URL: ")
    video_id=video_url[-11::]
    comments=get_comments(youtube,video_id)
    sentiments=Sentiment_analysis(comments)
    df=pd.DataFrame(comments)
    df.to_csv('Commentlist.csv',index=False,encoding='utf-8')
    print(f"Saved {len(comments)} comments")
    print(sentiments)
    plotSentiments(sentiments)
                