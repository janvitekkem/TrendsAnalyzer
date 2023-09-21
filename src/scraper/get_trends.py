from datetime import date
import snscrape.modules.twitter as sntwitter

#This function scrapes a batch of 100 tweets for the given trend

def retrive_tweets(trend, limit = 20, isVerified = False):
    print(f"Scrapper Started on {date.today()}")
    scraper = sntwitter.TwitterSearchScraper(trend)
    tweets: list[str] = [] 
    for i, tweet in enumerate(scraper.get_items(), 0):
        data = tweet.rawContent
        tweets.append(data)
        if i > limit:
            break
    return tweets

#end of fn














if __name__ == '__main__':
    retrive_tweets('#RakshaBandhan')