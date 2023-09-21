import functools
from pydantic import BaseModel
from src import scrap_hashtag
from src.utils.utils import prediction_to_sentiment, reducer
from src.analyzer.predict import predict_hashtag_sentiment
# from src.hugchat.analyzer import predict_sentiment
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI()

@app.get("/trend/{hashtag}/")
async def root(hashtag):
    print(hashtag)
    scrapped_tweets = scrap_hashtag(hashtag)
    aggregated_tweet = functools.reduce(reducer, scrapped_tweets)
    sentiment_prediction = predict_hashtag_sentiment(aggregated_tweet)
    sentiment = prediction_to_sentiment(sentiment_prediction)
    # zero_shot = predict_sentiment(aggregated_tweet)
    # print(zero_shot)
    return {"scrapped_tweets": scrapped_tweets, "sentiment": sentiment, "prediction": sentiment_prediction[0] }

app.mount("/static", StaticFiles(directory="./static", html=True), name="static")
