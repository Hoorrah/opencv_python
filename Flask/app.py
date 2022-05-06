from flask import Flask, redirect, url_for, render_template, request
import json
from flask import Flask, jsonify
from isbnlib import *
from urllib.request import urlopen
from json2html import *

# we need this function for Google API function
def remove(string):
	return string.replace(" ", "")

#this function use isbnlib for getting information by isbn of book
def isbnlibisbn(isbn, typ): #type1 = google book , type2 = wikipedia , type3 = openlibrary
	if (typ == 1):
		databook = meta(isbn, service='goob') #getting data from google service book and return it in databook
	if (typ == 2):
		databook = meta(isbn, service='wiki')#getting data from wikipedia and return it in databook
	if (typ == 3):
		databook = meta(isbn, service='openl')#getting data from open library and return it in databook
	book= '{"ISBN-13": "" ,"Title": "", "Authors":"" , "Publisher":"" , "Year": "", "Language": "", "PageCount": "", "PreviewLink": "", "Keywords": "" , "comment": "" ,"imageCover": "" , "Description":""}'
	book = json.loads(book) #change data to json value
	#write data to the book, to have same format for all the functions
	book["ISBN-13"] =databook["ISBN-13"]
	book["Title"] =databook["Title"]
	book["Authors"] =databook["Authors"]
	book["Publisher"] =databook["Publisher"]
	book["Year"] =databook["Year"]
	book["Language"] =databook["Language"]
	book["Description"] = desc(isbn)
	book["imageCover"] =cover(isbn)
	return book

###################################################################

#this function use isbnlib for getting information by title of book
def isbnlibtitle(title , typ): #type1 = google book , type2 = wikipedia , type3 = openlibrary
	#isbn = isbn_from_words(title) #find the isbn of the book by title of the book
	if (typ == 1):
		databook = meta(isbn, service='goob')#getting data from google service book and return it in databook
	if (typ == 2):
		databook = meta(isbn, service='wiki')#getting data from wikipedia and return it in databook
	if (typ == 3):
		databook = meta(isbn, service='openl')#getting data from open library and return it in databook
	book= '{"ISBN-13": "" ,"Title": "", "Authors":"" , "Publisher":"" , "Year": "", "Language": "", "PageCount": "", "PreviewLink": "", "Keywords": "" , "comment": "" ,"imageCover": "" , "Description":""}'
	book = json.loads(book) #change data to json value
	#write data to the book, to have same format for all the functions
	book["ISBN-13"] =databook["ISBN-13"]
	book["Title"] =databook["Title"]
	book["Authors"] =databook["Authors"]
	book["Publisher"] =databook["Publisher"]
	book["Year"] =databook["Year"]
	book["Language"] =databook["Language"]
	book["Description"] = desc(isbn)
	book["imageCover"] =cover(isbn)
	return book

###################################################################

def googleAPI(inpute , typ): #if type = 1 then input is isbn and if type = 0 the inpute is Title
	if (typ == 1):
		api = "https://www.googleapis.com/books/v1/volumes?q=isbn:" #api for searching by isbn of book
	if (typ == 0):
		api = "https://www.googleapis.com/books/v1/volumes?q=title:" #api for searching by title of the book
	# send a request and get a JSON response
	resp = urlopen(api + str(remove(inpute)))
	book= '{"ISBN-13": "" ,"Title": "", "Authors":"" , "Publisher":"" , "Year": "", "Language": "", "PageCount": "", "PreviewLink": "" , "comment": "" ,"imageCover": "" , "Description":""}'
	book = json.loads(book)
	# parse JSON into Python as a dictionary
	book_data = json.load(resp)
	volume_info = book_data["items"][0]["volumeInfo"]
	#because some of data are not available for some books we should write try for each of them
	try:
		book["Title"] = volume_info['title']
	except: pass
	try:
		book["Language"] = volume_info['language']
	except: pass
	try:
		author = volume_info["authors"]
		# practice with conditional expressions!
		prettify_author = author if len(author) > 1 else author[0]
		#display title, author, page count, publication date
		book["Authors"] = prettify_author
	except: pass
	try:
		book["Punlisher"] = volume_info['publisher']
	except: pass
	try:
		book["ISBN-13"] = volume_info['industryIdentifiers'][0]['identifier']
	except: pass
	try:
		book["PageCount"] = volume_info['pageCount']
	except: pass
	try:
		book["Year"] = volume_info['publishedDate']
	except: pass
	try:
		book["PreviewLink"] = volume_info['previewLink']
	except: pass
	try:
		book["Description"] = volume_info['description']
	except: pass
	return book
#################################################################################


app = Flask(__name__)

#first page of the html code link : http://127.0.0.1:5000/
@app.route("/" , methods=['GET', 'POST'])
def home():
	if request.method == "POST":
		#if click on the isbn button
		if request.form['submit_button'] == 'ISBN':
			isbn = request.form["nm"] #getting the inpute from the html code
			return redirect(url_for("isbn" , isbn = isbn)) # go to /isbn/<isbn> by isbn which get from inpute
		#if click on the isbn button
		elif request.form['submit_button'] == 'Title':
			title = request.form["nm"] #getting the inpute from the html code
			return redirect(url_for("title" , title = title)) # go to /title/<title> by title which get from inpute
	else:
		return render_template("choose.html")



@app.route('/isbn/<isbn>', methods=['GET', 'POST'])
def isbn(isbn):
	if request.method == "POST":
		#if click on the google book servive button go to the page GoogleServiceISBN
		if request.form['submit_button'] == 'Google Books service':
			return redirect(url_for("GoogleServiceISBN", inpute=isbn))
		#if click on the wikipedia.org button go to the page wikipediaISBN
		elif request.form['submit_button'] == 'wikipedia.org':
			return redirect(url_for("wikipediaISBN", inpute=isbn ))
		#if click on the OpenLibrary.org button go to the page OpenLibraryISBN
		elif request.form['submit_button'] == 'OpenLibrary.org':
			return redirect(url_for("OpenLibraryISBN", inpute=isbn))
		#if click on the Google Book API button go to the page GoogleAPIISBN
		elif request.form['submit_button'] == 'Google Books API':
			return redirect(url_for("GoogleAPIISBN", inpute=isbn))
	else:
		return render_template("source.html")



@app.route('/title/<title>', methods=['GET', 'POST'])
def title(title):
	if request.method == "POST":
		#if click on the google book servive button go to the page GoogleServiceTitle
		if request.form['submit_button'] == 'Google Books service':
			return redirect(url_for("GoogleServiceTitle", inpute=title))
		#if click on the wikipedia.org button go to the page wikipediaTitle
		elif request.form['submit_button'] == 'wikipedia.org':
			return redirect(url_for("wikipediaTitle", inpute=title))
		#if click on the OpenLibrary.org button go to the page OpenLibraryTitle
		elif request.form['submit_button'] == 'OpenLibrary.org':
			return redirect(url_for("OpenLibraryTitle", inpute=title))
		#if click on the Google Book API button go to the page GoogleAPITitle
		elif request.form['submit_button'] == 'Google Books API':
			return redirect(url_for("GoogleAPITitle", inpute=title))
	else:
		return render_template("source.html")

################################################################################

@app.route('/isbn/GoogleService/<inpute>' , methods=['GET', 'POST'])
def GoogleServiceISBN(inpute):
	try:
		# finding information of book by isbn from google book servise by using isbnlibisbn function
		isbn = inpute.strip()
		googlebook = isbnlibisbn(inpute , 1)
		table = json2html.convert(json = googlebook) #write data from json format to html table format
		return table
	except: # if data is invalid write the error message and let the user go to the first page by button
		if request.method == "POST":
			return redirect(url_for("home"))
		else:
			return render_template("invalid.html")



@app.route('/title/GoogleService/<inpute>' , methods=['GET', 'POST'])
def GoogleServiceTitle(inpute):
	try:
		#finding information of book by title from google book servise by using isbnlibisbn function
		title = inpute.strip()
		isbn = isbn_from_words(title) # finding isbn of the book from title ( by a function from isbnlib )
		googlebook = isbnlibisbn(isbn , 1)
		table = json2html.convert(json = googlebook)#write data from json format to html table format
		return table
	except: # if data is invalid write the error message and let the user go to the first page by button
		if request.method == "POST":
			return redirect(url_for("home"))
		else:
			return render_template("invalid.html")

################################################################################

@app.route('/isbn/wikipedia/<inpute>' , methods=['GET', 'POST'])
def wikipediaISBN(inpute):
	try:
		#finding information of book by isbn from wikipedia.org by using isbnlibisbn function
		isbn = inpute.strip()
		wikibook = isbnlibisbn(isbn , 2)
		table = json2html.convert(json = wikibook)#write data from json format to html table format
		return table
	except: # if data is invalid write the error message and let the user go to the first page by button
		if request.method == "POST":
			return redirect(url_for("home"))
		else:
			return render_template("invalid.html")



@app.route('/title/wikipedia/<inpute>' , methods=['GET', 'POST'])
def wikipediaTitle(inpute):
	try:
		#finding information of book by title from wikipedia.org by using isbnlibisbn function
		title = inpute.strip()
		isbn = isbn_from_words(title) # finding isbn of the book from title ( by a function from isbnlib )
		wikibook = isbnlibtitle(isbn , 2)
		table = json2html.convert(json = wikibook)#write data from json format to html table format
		return table
	except: # if data is invalid write the error message and let the user go to the first page by button
		if request.method == "POST":
			return redirect(url_for("home"))
		else:
			return render_template("invalid.html")

################################################################################

@app.route('/isbn/OpenLibrary/<inpute>' , methods=['GET', 'POST'])
def OpenLibraryISBN(inpute):
	try:
		#finding information of book by isbn from OpenLibrary.org by using isbnlibisbn function
		isbn = inpute.strip()
		openlibbook = isbnlibisbn(isbn , 3)
		table = json2html.convert(json = openlibbook)#write data from json format to html table format
		return table
	except: # if data is invalid write the error message and let the user go to the first page by button
		if request.method == "POST":
			return redirect(url_for("home"))
		else:
			return render_template("invalid.html")


@app.route('/title/OpenLibrary/<inpute>' , methods=['GET', 'POST'])
def OpenLibraryTitle(inpute):
	try:
		#finding information of book by title from OpenLibrary.org by using isbnlibisbn function
		title = inpute.strip()
		isbn = isbn_from_words(title) # finding isbn of the book from title ( by a function from isbnlib )
		openlibbook = isbnlibtitle(isbn , 3)
		table = json2html.convert(json = openlibbook)#write data from json format to html table format
		return table
	except: # if data is invalid write the error message and let the user go to the first page by button
		if request.method == "POST":
			return redirect(url_for("home"))
		else:
			return render_template("invalid.html")

################################################################################

@app.route('/isbn/GoogleAPI/<inpute>' , methods=['GET', 'POST'])
def GoogleAPIISBN(inpute):
	try:
		#finding information of book by isbn from Google book API by using googleAPI function
		isbn = inpute.strip()
		apibook = googleAPI(isbn, 1)
		table = json2html.convert(json = apibook )#write data from json format to html table format
		return table
	except: # if data is invalid write the error message and let the user go to the first page by button
		if request.method == "POST":
			return redirect(url_for("home"))
		else:
			return render_template("invalid.html")



@app.route('/title/GoogleAPI/<inpute>' , methods=['GET', 'POST'])
def GoogleAPITitle(inpute):
	try:
		#finding information of book by title from Google book API by using googleAPI function
		title = inpute.strip()
		isbn = isbn_from_words(title) # finding isbn of the book from title ( by a function from isbnlib )
		apibook = googleAPI(isbn, 2)
		table = json2html.convert(json = apibook )#write data from json format to html table format
		return table
	except: # if data is invalid write the error message and let the user go to the first page by button
		if request.method == "POST":
			return redirect(url_for("home"))
		else:
			return render_template("invalid.html")

################################################################################

if __name__ == "__main__":
	app.run(debug=True)
