import threading
import requests
from datetime import datetime
from transformers import pipeline
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def scrape_reddit_topics(search_query):
    """Scrape Reddit posts & comments for any search term (topic)."""
    url = f"https://www.reddit.com/search.json?q={search_query}&sort=relevance"
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/123.0.0.0 Safari/537.36'
        )
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()

    posts_with_comments = []
    for post in data['data']['children']:
        pd = post['data']
        cu = datetime.fromtimestamp(pd.get('created_utc', 0)).isoformat()
        comments_url = f"https://www.reddit.com{pd['permalink']}.json"
        cr = requests.get(comments_url, headers=headers)
        cr.raise_for_status()
        cdata = cr.json()
        comments = []
        if len(cdata) > 1:
            for com in cdata[1]['data']['children']:
                d = com['data']
                if 'body' in d:
                    comments.append({
                        'author': d.get('author', '[deleted]'),
                        'body':   d['body'],
                        'score':  d.get('score', 0),
                        'created_utc': datetime.fromtimestamp(d['created_utc']).isoformat()
                    })
        posts_with_comments.append({
            'title':        pd.get('title', ''),
            'author':       pd.get('author', '[deleted]'),
            'score':        pd.get('score', 0),
            'url':          pd.get('url', ''),
            'created_utc':  cu,
            'num_comments': pd.get('num_comments', 0),
            'selftext':     pd.get('selftext', ''),
            'comments':     comments
        })
    return posts_with_comments

def extract_posts_and_comments(data):
    """Flatten all post selftexts and comment bodies into a list of strings."""
    texts = []
    for e in data:
        if e.get('selftext'):
            texts.append(e['selftext'])
        for c in e.get('comments', []):
            if c.get('body'):
                texts.append(c['body'])
    return texts

def analyze_sentiment(texts):
    """Run the sentiment pipeline and count labels."""
    clf = pipeline(
        'sentiment-analysis',
        model='tabularisai/multilingual-sentiment-analysis'
    )
    counts = {k: 0 for k in (
        'Very Positive', 'Positive', 'Neutral', 'Negative', 'Very Negative'
    )}
    for t in texts:
        lbl = clf(t[:512])[0]['label']
        if lbl in counts:
            counts[lbl] += 1
    return counts

def plot_sentiment(counts):
    """Return a Matplotlib Figure for the sentiment pie chart."""
    labels = list(counts.keys())
    sizes  = [counts[l] for l in labels]
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']
    fig, ax = plt.subplots()
    ax.pie(sizes, colors=colors, labels=labels,
           autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    ax.set_title('Sentiment Analysis of Reddit Search Results')
    return fig

class RedditSentimentApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Reddit Sentiment Analyzer")
        self.geometry("800x600")
        self._build_ui()

    def _build_ui(self):
        frm = ttk.Frame(self); frm.pack(pady=10)
        ttk.Label(frm, text="Topic:").grid(row=0, column=0, padx=5)
        self.entry = ttk.Entry(frm, width=30); self.entry.grid(row=0, column=1)
        ttk.Button(frm, text="Analyze", command=self._on_click).grid(
            row=0, column=2, padx=5
        )
        self.chart_frame = ttk.Frame(self)
        self.chart_frame.pack(fill=tk.BOTH, expand=True)

    def _on_click(self):
        for w in self.chart_frame.winfo_children():
            w.destroy()
        ttk.Label(self.chart_frame, text="Loading…").pack(pady=20)
        threading.Thread(target=self._fetch_and_count, daemon=True).start()

    def _fetch_and_count(self):
        topic = self.entry.get().strip()
        if not topic:
            self.after(0, self._show_error, "Please enter a topic.")
            return

        try:
            data   = scrape_reddit_topics(topic)
            if not data:
                raise RuntimeError("No results returned.")
            texts  = extract_posts_and_comments(data)
            counts = analyze_sentiment(texts)
            # schedule plot creation on main thread
            self.after(0, self._show_plot, counts)
        except Exception as e:
            self.after(0, self._show_error, str(e))

    def _show_plot(self, counts):
        for w in self.chart_frame.winfo_children():
            w.destroy()
        fig    = plot_sentiment(counts)
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _show_error(self, msg):
        for w in self.chart_frame.winfo_children():
            w.destroy()
        ttk.Label(
            self.chart_frame, text=f"Error: {msg}", foreground="red"
        ).pack(pady=20)

if __name__ == "__main__":
    app = RedditSentimentApp()
    app.mainloop()
