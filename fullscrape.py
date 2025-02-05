import praw
import csv
import os
from dotenv import load_dotenv
load_dotenv()

CLIENT_ID = os.environ.get('CLIENT_ID')
CLIENT_SECRET = os.environ.get('CLIENT_SECRET')
USER_AGENT = os.environ.get('USER_AGENT')

def get_reddit_instance():
    return praw.Reddit(client_id=CLIENT_ID,
                    client_secret=CLIENT_SECRET,
                    user_agent=USER_AGENT)

def main():
    reddit = get_reddit_instance()
    subreddit = reddit.subreddit('python')
    # Retrieve top 50 posts over the past month
    posts = subreddit.top(time_filter='month', limit=50)

    with open('posts_with_timestamps.csv', mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['title', 'score', 'created_utc']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for post in posts:
            writer.writerow({
                'title': post.title,
                'score': post.score,
                'created_utc': post.created_utc
            })
    print("Data successfully written to posts_with_timestamps.csv")

if __name__ == "__main__":
    main()
