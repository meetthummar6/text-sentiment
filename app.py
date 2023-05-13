from flask import Flask,render_template,request
import pickle
import pandas as pd
import string
import re
import nltk

#loading the model
with open('model.pkl','rb') as f:
    model=pickle.load(f)

app=Flask(__name__)

@app.route("/",methods=["GET","POST"])
def index():
    if request.method == "POST":
        #get the form data
        text=request.form["text"]
        text=text.lower()
        stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
             'does', 'doing', 'down', 'during', 'each','few', 'for', 'from',
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're','s', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
             "youve", 'your', 'yours', 'yourself', 'yourselves']
        STOPWORDS = set(stopwordlist)
        def cleaning_stopwords(text):
            return " ".join([word for word in str(text).split() if word not in STOPWORDS])
        text=cleaning_stopwords(text)
        english_punctuations = string.punctuation
        punctuations_list = english_punctuations
        def cleaning_punctuations(text):
            translator = str.maketrans('', '', punctuations_list)
            return text.translate(translator)
        text=cleaning_punctuations(text)
        def cleaning_repeating_char(text):
            return re.sub(r'(.)1+', r'1', text)
        text=cleaning_repeating_char(text)
        def cleaning_numbers(data):
            return re.sub('[0-9]+', '', data)
        text=cleaning_numbers(text)
        from nltk.tokenize import RegexpTokenizer
        tokenizer = RegexpTokenizer(r'\w+')
        text=tokenizer.tokenize(text)
        st = nltk.PorterStemmer()
        def stemming_on_text(data):
            text = [st.stem(word) for word in data]
            return text
        lm = nltk.WordNetLemmatizer()
        def lemmatizer_on_text(data):
            text = [lm.lemmatize(word) for word in data]
            return text
        textstem=stemming_on_text(text)
        textlem=lemmatizer_on_text(textstem)
        #maybe we need to pickle the vectorizer as well need to work on that tommorow
        vectorizer=pickle.load(open("vectoriser.pkl","rb"))
        text=vectorizer.transform(textlem)
        prediction=model.predict(text)
        print(prediction)
        return render_template("index.html",prediction=prediction[0])
    return render_template("index.html",prediction=3)


if __name__ == '__main__':
    app.run(debug=True)