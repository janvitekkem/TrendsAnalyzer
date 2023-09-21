# from hugchat import hugchat
# import json

# chatbot = hugchat.ChatBot(cookie_path="cookies.json")  # or cookies=[...]


# def predict_sentiment(tweets):
#     prompt = f"""
#                 Your task is to help predict the sentiment of tweets, which is delimited by backticks.
#                 Tweets: `{tweets}`
#                 Output the answer as JSON.
#                 Use the following format.
#                 sentiment_probablity: <probablity of sentiment (0-1) shown in the tweets>
#                 sentiment: < sentiment shown in the tweets >
#                 emotions: <emotions>
#                 summary: <summarise the discussion thats happening in the tweets>
                
#             """
#     resp =chatbot.chat(prompt)
#     return json.loads(resp)

