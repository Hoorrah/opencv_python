
from isbntools.app import *

#getting isbn from name of book
get_isbn = isbn_from_words("me befor you")

#getting book data from isbn
det_book = registry.bibformatters['labels'](meta("978-963-528-942-4"))
#det_book = registry.bibformatters['json'](meta("9780718181185"))
print(get_isbn)
print(det_book)
