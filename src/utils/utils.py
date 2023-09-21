def prediction_to_sentiment(y_hat: float) -> str:
    if y_hat > 0.5:
        return'Positive sentiment'
    elif y_hat==0.5:
        return 'Neutral sentiment'
    else: 
        return 'Negative sentiment'
    
def reducer(tweet_a,tweet_b) -> str:
    return tweet_a + tweet_b