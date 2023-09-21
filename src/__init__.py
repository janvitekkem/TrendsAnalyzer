import src.scraper.get_trends as scraper
import src.preprocessor.pre as preprocessor
import functools

def scrap_hashtag(trend_hashtag):
    tweet_content_list: list[str] = scraper.retrive_tweets(trend_hashtag)
    return tweet_content_list
