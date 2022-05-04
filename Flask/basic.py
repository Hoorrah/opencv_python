
import json
from flask import Flask, jsonify
from isbnlib import *
from urllib.request import urlopen

def remove(string):
	return string.replace(" ", "")

def googleisbn(isbn):
	googlebook = meta(isbn, service='goob')
	book= '{"ISBN-13": "" ,"Title": "", "Authors":"" , "Publisher":"" , "Year": "", "Language": "", "PageCount": "", "PreviewLink": "", "Keywords": "" , "comment": "" ,"imageCover": "" , "Description":""}'
	book = json.loads(book)
	book["ISBN-13"] =googlebook["ISBN-13"]
	book["Title"] =googlebook["Title"]
	book["Authors"] =googlebook["Authors"]
	book["Publisher"] =googlebook["Publisher"]
	book["Year"] =googlebook["Year"]
	book["language"] =googlebook["Language"]
	book["Description"] = desc(isbn)
	book["imageCover"] =cover(isbn)
	return book

##################################################################

def wikipediaisbn(isbn):
	wikipedia = meta(get_isbn, service='wiki')
	book= '{"ISBN-13": "" ,"Title": "", "Authors":"" , "Publisher":"" , "Year": "", "Language": "", "PageCount": "", "PreviewLink": "", "Keywords": "" , "comment": "" ,"imageCover": "" , "Description":""}'
	book = json.loads(book)
	book["ISBN-13"] =wikipedia["ISBN-13"]
	book["Title"] =wikipedia["Title"]
	book["Authors"] =wikipedia["Authors"]
	book["Publisher"] =wikipedia["Publisher"]
	book["Year"] =wikipedia["Year"]
	book["language"] =wikipedia["Language"]
	book["Description"] = desc(isbn)
	book["imageCover"] =cover(isbn)
	return book
##################################################################

def openlibisbn(isbn):
	openlibrary = meta(get_isbn, service='openl')
	book= '{"ISBN-13": "" ,"Title": "", "Authors":"" , "Publisher":"" , "Year": "", "Language": "", "PageCount": "", "PreviewLink": "", "Keywords": "" , "comment": "" ,"imageCover": "" , "Description":""}'
	book = json.loads(book)
	book["ISBN-13"] =openlibrary["ISBN-13"]
	book["Title"] =openlibrary["Title"]
	book["Authors"] =openlibrary["Authors"]
	book["Publisher"] =openlibrary["Publisher"]
	book["Year"] =openlibrary["Year"]
	book["language"] =openlibrary["Language"]
	book["Description"] = desc(isbn)
	book["imageCover"] =cover(isbn)
	return book

###################################################################

def googletitle(title):
	isbn = isbn_from_words(title)
	googlebook = meta(isbn, service='goob')
	book= '{"ISBN-13": "" ,"Title": "", "Authors":"" , "Publisher":"" , "Year": "", "Language": "", "PageCount": "", "PreviewLink": "", "Keywords": "" , "comment": "" ,"imageCover": "" , "Description":""}'
	book = json.loads(book)
	book["ISBN-13"] =googlebook["ISBN-13"]
	book["Title"] =googlebook["Title"]
	book["Authors"] =googlebook["Authors"]
	book["Publisher"] =googlebook["Publisher"]
	book["Year"] =googlebook["Year"]
	book["language"] =googlebook["Language"]
	book["Description"] = desc(isbn)
	book["imageCover"] =cover(isbn)
	return book

##################################################################

def wikipediatitle(title):
	isbn = isbn_from_words(title)
	wikipedia = meta(get_isbn, service='wiki')
	book= '{"ISBN-13": "" ,"Title": "", "Authors":"" , "Publisher":"" , "Year": "", "Language": "", "PageCount": "", "PreviewLink": "", "Keywords": "" , "comment": "" ,"imageCover": "" , "Description":""}'
	book = json.loads(book)
	book["ISBN-13"] =wikipedia["ISBN-13"]
	book["Title"] =wikipedia["Title"]
	book["Authors"] =wikipedia["Authors"]
	book["Publisher"] =wikipedia["Publisher"]
	book["Year"] =wikipedia["Year"]
	book["language"] =wikipedia["Language"]
	book["Description"] = desc(isbn)
	book["imageCover"] =cover(isbn)
	return book
##################################################################

def openlibtitle(title):
	isbn = isbn_from_words(title)
	openlibrary = meta(get_isbn, service='openl')
	book= '{"ISBN-13": "" ,"Title": "", "Authors":"" , "Publisher":"" , "Year": "", "Language": "", "PageCount": "", "PreviewLink": "", "Keywords": "" , "comment": "" ,"imageCover": "" , "Description":""}'
	book = json.loads(book)
	book["ISBN-13"] =openlibrary["ISBN-13"]
	book["Title"] =openlibrary["Title"]
	book["Authors"] =openlibrary["Authors"]
	book["Publisher"] =openlibrary["Publisher"]
	book["Year"] =openlibrary["Year"]
	book["language"] =openlibrary["Language"]
	book["Description"] = desc(isbn)
	book["imageCover"] =cover(isbn)
	return book

###################################################################

def googleAPI(inpute , type): #if type = 1 then input is isbn and if type = 0 the inpute is Title
	if (type == 1):
		api = "https://www.googleapis.com/books/v1/volumes?q=isbn:"
	if (type == 0):
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

inputtype = input("Write tyepe of you are searching for book: ISBN --> i / Title -->t : ")

if (inputtype == "i"):
	get_isbn =  input("Write the ISBN:").strip()

elif (inputtype == "t"):
	inputdata =  input("Write Title of the book:").strip()
	get_isbn = isbn_from_words(inputdata)

else:
	print("invalid input")


app = Flask(__name__)

@app.route('/') #create url
def index():#the funvtion return the tring to the url
	return "Book information"

@app.route('/<isbn>')
def getbyisbn(isbn):
	googlebook = googleisbn(get_isbn)
	wikibook = wikipediaisbn(get_isbn)
	openlibbook = openlibisbn(get_isbn)
	apibook = googleAPI(get_isbn, 1)
	strresults = "Google :\n" + str(googlebook) +"wikipedia :\n" + str(wikibook) + "OpenLibrary :\n" + str(openlibbook) +"Google API : \n" +str (apibook)
	return strresults

@app.route('/<title>')
def getbytitle(title):
	googlebook = googletitle(get_isbn)
	wikibook = wikipediatitle(get_isbn)
	openlibbook = openlibtitle(get_isbn)
	apibook = googleAPI(get_isbn, 0)
	strresults = "Google :\n" + str(googlebook) +"wikipedia :\n" + str(wikibook) + "OpenLibrary :\n" + str(openlibbook) +"Google API \n" +str (apibook)
	apibook = googleAPI(get_isbn, 0)
	return strresults


if __name__ == "__main__":
	app.run(debug=True)
