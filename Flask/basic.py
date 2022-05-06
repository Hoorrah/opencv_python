from flask import Flask, redirect, url_for, render_template, request
import json
from flask import Flask, jsonify
from isbnlib import *
from urllib.request import urlopen
from json2html import *

def remove(string):
    return string.replace(" ", "")

def isbnlibisbn(isbn, typ): #type1 = google book , type2 = wikipedia , type3 = openlibrary
    if (typ == 1):
        databook = meta(isbn, service='goob')
        print(type(databook))
    if (typ == 2):
        databook = meta(isbn, service='wiki')
    if (typ == 3):
        databook = meta(isbn, service='openl')
    book= '{"ISBN-13": "" ,"Title": "", "Authors":"" , "Publisher":"" , "Year": "", "Language": "", "PageCount": "", "PreviewLink": "", "Keywords": "" , "comment": "" ,"imageCover": "" , "Description":""}'
    book = json.loads(book)
    book["ISBN-13"] =databook["ISBN-13"]
    book["Title"] =databook["Title"]
    book["Authors"] =databook["Authors"]
    book["Publisher"] =databook["Publisher"]
    book["Year"] =databook["Year"]
    book["language"] =databook["Language"]
    book["Description"] = desc(isbn)
    book["imageCover"] =cover(isbn)
    return book

###################################################################

def isbnlibtitle(title , typ): #type1 = google book , type2 = wikipedia , type3 = openlibrary
    isbn = isbn_from_words(title)
    if (typ == 1):
        databook = meta(isbn, service='goob')
    if (typ == 2):
        databook = meta(isbn, service='wiki')
    if (typ == 3):
        databook = meta(isbn, service='openl')
    book= '{"ISBN-13": "" ,"Title": "", "Authors":"" , "Publisher":"" , "Year": "", "Language": "", "PageCount": "", "PreviewLink": "", "Keywords": "" , "comment": "" ,"imageCover": "" , "Description":""}'
    book = json.loads(book)
    book["ISBN-13"] =databook["ISBN-13"]
    book["Title"] =databook["Title"]
    book["Authors"] =databook["Authors"]
    book["Publisher"] =databook["Publisher"]
    book["Year"] =databook["Year"]
    book["language"] =databook["Language"]
    book["Description"] = desc(isbn)
    book["imageCover"] =cover(isbn)
    return book

###################################################################

def googleAPI(inpute , typ): #if type = 1 then input is isbn and if type = 0 the inpute is Title
    if (typ == 1):
        api = "https://www.googleapis.com/books/v1/volumes?q=isbn:"
    if (typ == 0):
        api = "https://www.googleapis.com/books/v1/volumes?q=title:"
    # send a request and get a JSON response
    resp = urlopen(api + str(remove(inpute)))
    book= '{"ISBN-13": "" ,"Title": "", "Authors":"" , "Publisher":"" , "Year": "", "Language": "", "PageCount": "", "PreviewLink": "" , "comment": "" ,"imageCover": "" , "Description":""}'
    book = json.loads(book)
    # parse JSON into Python as a dictionary
    book_data = json.load(resp)
    volume_info = book_data["items"][0]["volumeInfo"]
    try:
        book["Title"] = {volume_info['title']}
    except: pass
    try:
        book["Language"] = {volume_info['language']}
    except: pass
    try:
        author = volume_info["authors"]
        # practice with conditional expressions!
        prettify_author = author if len(author) > 1 else author[0]
        #display title, author, page count, publication date
        book["Authors"] = {prettify_author}
    except: pass
    try:
        book["Punlisher"] = {volume_info['publisher']}
    except: pass
    try:
        book["ISBN-13"] = {volume_info['industryIdentifiers'][0]['identifier']}
    except: pass
    try:
        book["PageCount"] = {volume_info['pageCount']}
    except: pass
    try:
        book["Year"] = {volume_info['publishedDate']}
    except: pass
    try:
        book["PreviewLink"] = {volume_info['previewLink']}
    except: pass
    try:
        book["Description"] = {volume_info['description']}
    except: pass
    return book
#######################################################################################


app = Flask(__name__)

@app.route("/" , methods=['GET', 'POST'])
def home():
    if request.method == "POST":
        if request.form['submit_button'] == 'ISBN':
            isbn = request.form["nm"]
            return redirect(url_for("isbn" , isbn = isbn))
        elif request.form['submit_button'] == 'Title':
            title = request.form["nm"]
            return redirect(url_for("title" , title = title))
    else:
        return render_template("choose.html")



@app.route('/<isbn>', methods=['GET', 'POST'])
def isbn(isbn):
    if request.method == "POST":
        if request.form['submit_button'] == 'Google Books service':
            return redirect(url_for("GoogleServiceISBN", inpute=isbn))
        elif request.form['submit_button'] == 'wikipedia.org':
            return redirect(url_for("wikipedia", inpute=isbn , typ = 1))
        elif request.form['submit_button'] == 'OpenLibrary.org':
            return redirect(url_for("OpenLibrary", inpute=isbn , typ = 1))
        elif request.form['submit_button'] == 'Google Books API':
            return redirect(url_for("GoogleAPI", inpute=isbn , typ = 1))
    else:
        return render_template("source.html")



@app.route('/<title>', methods=['GET', 'POST'])
def title(title):
    if request.method == "POST":
        if request.form['submit_button'] == 'Google Books service':
            return redirect(url_for("GoogleBooksService", inpute=title))
        elif request.form['submit_button'] == 'wikipedia.org':
            return redirect(url_for("wikipedia", inpute=title , typ = 2))
        elif request.form['submit_button'] == 'OpenLibrary.org':
            return redirect(url_for("OpenLibrary", inpute=title , typ = 2))
        elif request.form['submit_button'] == 'Google Books API':
            return redirect(url_for("GoogleAPI", inpute=title , typ = 2))
    else:
        return render_template("source.html")



@app.route('/isbn/<inpute>' , methods=['GET', 'POST'])
def GoogleServiceISBN(inpute):
    isbn = inpute.strip()
    googlebook = isbnlibisbn(inpute , 1)
    tableisbn = json2html.convert(json = googlebook)
    return tableisbn

@app.route('/title/<inpute>' , methods=['GET', 'POST'])
def GoogleServiceTitle(inpute):
    title = inpute.strip()
    isbn = isbn_from_words(title)
    googlebook = isbnlibisbn(inpute , 1)
    tableisbn = json2html.convert(json = googlebook)
    return tableisbn

@app.route('/isbn/<inpute>' , methods=['GET', 'POST'])
def wikipediaISBN(inpute):
    isbn = inpute.strip()
    wikibook = isbnlibisbn(isbn , 2)
    tableisbn = json2html.convert(json = wikibook)
    return tableisbn

@app.route('/title/<inpute>' , methods=['GET', 'POST'])
def wikipediaTitle(inpute):
    title = inpute.strip()
    isbn = isbn_from_words(title)
    wikibook = isbnlibtitle(isbn , 2)
    tableisbn = json2html.convert(json = wikibook)
    return tabletitle



@app.route('/<inpute>/<typ>' , methods=['GET', 'POST'])
def OpenLibrary(inpute , typ ):
    if (typ == 1):
        isbn = inpute.strip()
        openlibbook = isbnlibisbn(isbn , 3)
        tableisbn = json2html.convert(json = openlibbook)
        return tableisbn
    if (typ == 2):
        title = inpute.strip()
        isbn = isbn_from_words(title)
        openlibbook = isbnlibtitle(isbn , 3)
        tableisbn = json2html.convert(json = openlibbook)
        return tabletitle



@app.route('/<inpute>/<typ>' , methods=['GET', 'POST'])
def GoogleAPI(inpute , typ ):
    if (typ == 1):
        isbn = inpute.strip()
        apibook = googleAPI(isbn, 1)
        tableisbn = json2html.convert(json = apibook )
        return tableisbn
    if (typ == 2):
        title = inpute.strip()
        isbn = isbn_from_words(title)
        apibook = googleAPI(isbn, 2)
        tableisbn = json2html.convert(json = apibook )
        return tabletitle



if __name__ == "__main__":
    app.run(debug=True)
