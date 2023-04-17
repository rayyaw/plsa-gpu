#!/usr/bin/python3

import urllib.error
import urllib.request

# Download books into the books/ folder. Keep going until no books are found (error 404).
# The first 10 books have a different format
count = 46455

while True:
    try:
        book_url = urllib.request.urlopen("https://www.gutenberg.org/ebooks/{count}.txt.utf-8".format(count=count), timeout=10)

        # Section out between *** START and *** END
        book = str(book_url.read(), "utf-8")

        book_split = book.split("*** START")

        # Get the author and title - we need these for the output file name
        header = book_split[0]
        book_split = book_split[1].split("*** END")
        
        # The footer is unused
        main_text = book_split[0]

        title_loc = header.find("Title: ")
        author_loc = header.find("Author: ")

        # If not found, use _, _ 
        title = '_'
        author = '_'

        if (title_loc != -1):
            title_loc += len("Title: ")
            title = header[title_loc:header.find("\n", title_loc)].strip()

        if (author_loc != -1):
            author_loc += len("Author: ")
            author = header[author_loc:header.find("\n", author_loc)].strip()

        fname = "books/" + str(count) + " - " + title + ", " + author + ".txt"

        # Save the book to a file (no preprocessing)
        with open(fname, "w+") as f:
             f.write(main_text)

        print("Downloaded", count, "books (of about 70,000)")
        count += 1
    except urllib.error.HTTPError as e:
        # If 404, we've read all the ebooks
        if (e.code == 404 and count > 70000):
            print("breaking at count", count)
            break

        print("skipping book", count)
        count += 1
    
    # FileNotFoundError happens when the book name has invalid characters (like /) in it
    except (FileNotFoundError, urllib.error.URLError, IndexError, OSError) as e:
        print("skipping book", count)
        count += 1
        continue
    #except BaseException as e:
    #    print(e)
