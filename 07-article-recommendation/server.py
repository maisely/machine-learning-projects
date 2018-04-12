from flask import Flask, render_template
from doc2vec import *
import sys


app = Flask(__name__)

@app.route("/")
def articles():
    """Show a list of article titles"""

    return render_template('articles.html', articlelist=articles)


@app.route("/article/<topic>/<filename>")
def article(topic,filename):
    """
    Show an article with relative path filename. Assumes the BBC structure of
    topic/filename.txt so our URLs follow that.
    """
    docs = topic + "/" + filename
    for article in articles:
        if article[0] == docs:
            title = article[1]
            content = article[2].split('\n')
            seealso = recommended(article, articles, 5)

    return render_template('article.html', articlelist=articles,
                                           title=title,
                                           content=content,
                                           seealso=seealso)

# initialization

i = sys.argv.index('server:app')
glove_filename = sys.argv[i+1]
articles_dirname = sys.argv[i+2]

gloves = load_glove(glove_filename)
articles = load_articles(articles_dirname, gloves)
