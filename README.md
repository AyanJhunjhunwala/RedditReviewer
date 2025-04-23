# RedditReviewer
# Why I made this

- Reddit had their APIs shift to a paid version and this circumvents that leaving the user to webscrape without spending 10 cents a call
- For product, reviews and general opinons, Reddit is consistently the best mix of opinons
- It saves me time from having to research broad and general topics

# What I used

- Data & NLP & Visualization: uses requests (get(), raise_for_status()) to fetch Reddit data; transformers.pipeline('sentiment-analysis') for sentiment; wordcloud.generate() with nltk stopwords for text filtering; and matplotlib.pyplot (subplots(), pie(), Circle(), legend(), axis(), set_title()) for plotting.
- Background & GUI: uses threading.Thread for off-UI processing; and tkinter/ttk (with a custom Canvas loader and styled Frame) plus PIL.ImageTk for the interactive interface.

https://github.com/user-attachments/assets/aa4a8031-031a-4698-a247-62c6751382c5

