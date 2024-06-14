import pandas as pd
import praw

client_id = "nSF6G2ITbnI7gn_-6fbEKw"
client_secret = "wFQsMrUvmxmXRTQvxLcgecdDzM_wpA"
user_agent = "Crypto/0.1 by BerryOk8646"

reddit = praw.Reddit(
    client_id=client_id, client_secret=client_secret, user_agent=user_agent
)


def _fetch_reddit_posts(subreddit, query, limit=100):
    subreddit = reddit.subreddit(subreddit)
    posts = subreddit.search(query, limit=limit)
    post_list = [[post.created_utc, post.title, post.selftext] for post in posts]
    df = pd.DataFrame(post_list, columns=["timestamp", "title", "text"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    return df


def process_reddit_posts() -> pd.DataFrame:
    subreddit = "cryptocurrency"
    query = "Bitcoin"
    limit = 100  # Number of posts to fetch
    reddit_df = _fetch_reddit_posts(subreddit, query, limit)
    # reddit_df.to_csv('data/bitcoin_reddit_posts.csv', index=False)
    return reddit_df
