
# This function extracts top referenced urls to this specific trend
def extract_top_urls(tweets):
    top_urls = {}
    for tweet in tweets:
        for url in tweet.urls:
            if url in top_urls.keys():
                top_urls[url] += 1
            else:
                top_urls[url] = 1
    x = {k: v for k, v in sorted(top_urls.items(), key=lambda item: item[1], reverse=True)}
    return dict(list(x.items())[:10])


# This function extracts top mentions to this specific trend
def extract_top_mentions(tweets):
    top_mentions = {}
    for tweet in tweets:
        for mention in tweet.mentions:
            if mention.get("screen_name", None) in top_mentions.keys():
                top_mentions[mention.get("screen_name", None)] += 1
            else:
                top_mentions[mention.get("screen_name", None)] = 1
    x = {k: v for k, v in sorted(top_mentions.items(), key=lambda item: item[1], reverse=True)}
    return dict(list(x.items())[:10])


# This function extracts other top hashtags to this specific trend
def extract_top_hashtags(tweets):
    top_hashtags = {}
    for tweet in tweets:
        for hashtag in tweet.hashtags:
            if hashtag in top_hashtags.keys():
                top_hashtags[hashtag] += 1
            else:
                top_hashtags[hashtag] = 1
    x = {k: v for k, v in sorted(top_hashtags.items(), key=lambda item: item[1], reverse=True)}
    return dict(list(x.items())[:10])
