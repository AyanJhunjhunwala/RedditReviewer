import tkinter as tk
from tkinter import ttk
from reddit_sentiment import scrape_reddit_topics, extract_posts_and_comments, analyze_sentiment, plot_sentiment
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class RedditSentimentApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Reddit Sentiment Analyzer")
        self.geometry("800x600")
        self.create_widgets()

    def create_widgets(self):
        frame = ttk.Frame(self)
        frame.pack(pady=10)
        ttk.Label(frame, text="Subreddit:").grid(row=0, column=0, padx=5)
        self.entry = ttk.Entry(frame, width=30)
        self.entry.grid(row=0, column=1, padx=5)
        ttk.Button(frame, text="Analyze", command=self.run_analysis).grid(row=0, column=2, padx=5)
        self.chart_frame = ttk.Frame(self)
        self.chart_frame.pack(fill=tk.BOTH, expand=True)

    def run_analysis(self):
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        subreddit = self.entry.get()
        data = scrape_reddit_topics(subreddit)
        if not data:
            ttk.Label(self.chart_frame, text="Error fetching data.").pack()
            return
        texts = extract_posts_and_comments(data)
        counts = analyze_sentiment(texts)
        fig = plot_sentiment(counts)
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    app = RedditSentimentApp()
    app.mainloop()
