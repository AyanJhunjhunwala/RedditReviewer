import requests
import json
from datetime import datetime
from transformers import pipeline
import matplotlib.pyplot as plt

def scrape_reddit_topics(search_query):
    """
    Scrape Reddit posts and comments based on a search query.
    Returns a list of dicts with post data and comments.
    """
    url = f"https://www.reddit.com/search.json?q={search_query}&sort=relevance"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/123.0.0.0 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        posts_with_comments = []
        for post in data['data']['children']:
            post_data = post['data']
            comments_url = f"https://www.reddit.com{post_data['permalink']}.json"
            comments_resp = requests.get(comments_url, headers=headers)
            comments_data = comments_resp.json()
            comments = []
            if len(comments_data) > 1:
                for comment in comments_data[1]['data']['children']:
                    cdata = comment['data']
                    if 'body' in cdata:
                        comments.append({
                            'author': cdata.get('author', '[deleted]'),
                            'body': cdata['body'],
                            'score': cdata.get('score', 0),
                            'created_utc': datetime.fromtimestamp(cdata['created_utc']).isoformat()
                        })
            posts_with_comments.append({
                'title': post_data.get('title', ''),
                'author': post_data.get('author', '[deleted]'),
                'score': post_data.get('score', 0),
                'url': post_data.get('url', ''),
                'created_utc': datetime.fromtimestamp(post_data.get('created_utc', 0)).isoformat(),
                'num_comments': post_data.get('num_comments', 0),
                'selftext': post_data.get('selftext', ''),
                'comments': comments
            })
        return posts_with_comments
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

def extract_posts_and_comments(json_obj):
    """Extract text content from posts and comments."""
    texts = []
    for entry in json_obj:
        if entry.get('selftext'):
            texts.append(entry['selftext'])
        for comment in entry.get('comments', []):
            if comment.get('body'):
                texts.append(comment['body'])
    return texts

def analyze_sentiment(texts):
    """Perform sentiment analysis on a list of texts."""
    classifier = pipeline('sentiment-analysis', model='tabularisai/multilingual-sentiment-analysis')
    labels = []
    for text in texts:
        truncated = text[:512]
        result = classifier(truncated)[0]['label']
        labels.append(result)
    counts = {
        'Very Positive': 0,
        'Positive': 0,
        'Neutral': 0,
        'Negative': 0,
        'Very Negative': 0
    }
    for label in labels:
        if label in counts:
            counts[label] += 1
    return counts

def plot_sentiment(counts):
    """Generate a pie chart for sentiment counts."""
    labels = list(counts.keys())
    sizes = [counts[key] for key in labels]
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']
    fig, ax = plt.subplots()
    ax.pie(sizes, colors=colors, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    plt.title('Sentiment Analysis of Reddit Posts and Comments')
    return fig

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        subreddit = sys.argv[1]
    else:
        subreddit = input("Enter subreddit to scrape: ")
    results = scrape_reddit_topics(subreddit)
    if results:
        texts = extract_posts_and_comments(results)
        counts = analyze_sentiment(texts)
        fig = plot_sentiment(counts)
        plt.show()
