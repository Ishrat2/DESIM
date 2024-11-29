#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 13:09:20 2024

@author: ishratvasid
"""

# Install FedTools and transformers packages
# pip install FedTools
# pip install transformers
# pip install yfinance
# import nltk
# nltk.download('punkt')

import pandas as pd
from FedTools import FederalReserveMins
from nltk import tokenize
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import yfinance as yf
import matplotlib.pyplot as plt

# Function to split a long sentence (>500) into manageable chunks
def split_into_chunks(text, max_length=500):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        word_length = len(word) + 1  # Account for a space or punctuation
        if current_length + word_length > max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(word)
        current_length += word_length

    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def process_long_sentence(sentence):
    chunks = split_into_chunks(sentence, max_length=500)
    results = [nlp(chunk) for chunk in chunks]  # Process each chunk separately
    
    # Combine results 
    aggregated_result = {
        "label": max(results, key=lambda x: x[0]['score'])[0]['label'],  # Most confident label
        "score": sum([x[0]['score'] for x in results]) / len(results)   # Average confidence score
    }
    return aggregated_result

# Define the details of scrapping
dataset = FederalReserveMins().find_minutes()["2019":"2023"]

# Import the pretrained classifer and tokenizer of Bert from FinBert online source
finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

# Build up the NLP pipeline with the pretrained classifer model and tokenizer from above
nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer, device=0)

date_list = dataset.index.tolist()
fomc_sentiments = dataset
for date in date_list:
    sentence_list = tokenize.sent_tokenize(dataset.loc[date,'Federal_Reserve_Mins'])
    positive_sent=0
    for sentence in sentence_list:
        if len(sentence)<=500: 
            nlp_result = pd.DataFrame(nlp(sentence))
        else:
            nlp_result = pd.DataFrame([process_long_sentence(sentence)])
        positive_sent += 1 if nlp_result.loc[0,"label"] == "Positive" else -1 if nlp_result.loc[0,"label"] == "Negative" else 0
        Sentiment_Score = positive_sent/len(sentence_list)
        fomc_sentiments.loc[date,"Sentiment_Score"]=Sentiment_Score

fomc_sentiments['Date'] = fomc_sentiments.index
sp500_data = yf.download('^GSPC', start='2018-12-31', end='2023-12-31') #ETF SPY
sp500_data['Return'] = sp500_data['Adj Close'].pct_change()
sp500_data['Date'] = sp500_data.index

results = []
for _, row in fomc_sentiments.iterrows():
    event_date = row['Date']
    sentiment_score = row['Sentiment_Score']
    
    # Filter S&P 500 data for the event window
    pre_window_start = event_date  - pd.tseries.offsets.BDay(4)
    post_window_end = event_date + pd.tseries.offsets.BDay(5)
    pre_window_data = sp500_data[(sp500_data['Date'] >= pre_window_start) & (sp500_data['Date'] <= event_date)].copy()
    post_window_data = sp500_data[(sp500_data['Date'] > event_date) & (sp500_data['Date'] <= post_window_end)].copy()
    
    # Calculate Cumulative Return
    results.append({
        'Event_Date': event_date,
        'Sentiment_Score': sentiment_score,
        'Cumulative_Return_Pre': [(1 + pre_window_data['Return']).cumprod() - 1][0][-1],
        'Cumulative_Return_Post': [(1 + post_window_data['Return']).cumprod() - 1][0][-1]
    })
results_df = pd.DataFrame(results)

# Compare Positive and Negative Sentiment Impact
positive_sentiment = results_df[results_df['Sentiment_Score'] >= 0]
negative_sentiment = results_df[results_df['Sentiment_Score'] < 0]

# Average Returns
print("Average Cumulative Return Pre-Event (Positive Sentiment):", positive_sentiment['Cumulative_Return_Pre'].mean())
print("Average Cumulative Return Post-Event (Positive Sentiment):", positive_sentiment['Cumulative_Return_Post'].mean())
print("Average Cumulative Return Pre-Event (Negative Sentiment):", negative_sentiment['Cumulative_Return_Pre'].mean())
print("Average Cumulative Return Post-Event (Negative Sentiment):", negative_sentiment['Cumulative_Return_Post'].mean())

# Visualization
plt.figure(figsize=(10, 6))
plt.bar(['Positive Pre', 'Positive Post', 'Negative Pre', 'Negative Post'], 
        [positive_sentiment['Cumulative_Return_Pre'].mean(), 
         positive_sentiment['Cumulative_Return_Post'].mean(), 
         negative_sentiment['Cumulative_Return_Pre'].mean(), 
         negative_sentiment['Cumulative_Return_Post'].mean()],
        color=['green', 'green', 'red', 'red'])
plt.title('Impact of Sentiment on S&P 500 Cumulative Returns')
plt.ylabel('Average Cumulative Return')
plt.show()

# Does the sentiment of the meeting minutes correlated with the market?
fig, ax1 = plt.subplots(figsize=(15, 7))

color = 'blue'
ax1.set_ylabel('SP500', color=color)
ax1.plot(sp500_data['Adj Close'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'orange'
ax2.set_ylabel('FOMC Sentiment Score', color=color)
ax2.plot(fomc_sentiments["Sentiment_Score"], color=color)
ax2.tick_params(axis='y', labelcolor=color)
plt.show()