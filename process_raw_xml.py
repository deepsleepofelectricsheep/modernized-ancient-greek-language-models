# This script processes the raw .xml files in the canonical greek lit repo, 
# purges them of their xml scaffolding, and stores the text body as .txt 
# files.
import os
import re
from glob import glob
from bs4 import BeautifulSoup


def main():
    exclude = [
        "This pointer", 
        "Keyboarding", 
        "professional data entry", 
        "The following",
        "optical",
        "Notes"
    ]

    authors = []

    # Create subfolder if it does not exist
    if not os.path.exists("data/text"):
        os.makedirs("data/text")

    # Iterate over all the relevant .xml files, and extract the raw text, 
    # the title and the author, and store each as a text file
    idx = 0
    for f in glob(
        "data/xml/**/*grc*.xml", 
        recursive=True
    ):
        if "__cts__" not in f and "INTF" not in f:
            with open(f, "r", errors="ignore", encoding="utf-8") as g:
                x = g.read()

            soup = BeautifulSoup(x, "xml")

            text_tags = soup.find_all(["l", "p"])
            title_tags = soup.find_all("title")
            author_tags = soup.find_all("author")
            if len(author_tags) == 0:
                author_tags = soup.find_all("docAuthor")

            # We will introduce hacky logic to determine when the 
            # author changes from one doc to the next to reset our idx for 
            # the filename
            authors.append(author_tags[0].text)
            if len(authors) > 1:
                if authors[-1] != authors[-2]:
                    idx = 0

            fname = f"{author_tags[0].text}-{idx}"
            fname = fname.replace('\n', '').replace('\t', '')
            text = f"{title_tags[0].text} by {author_tags[0].text}" + " \n " \
                + " \n ".join(el.text for el in text_tags if not any(w in el.text for w in exclude))
            
            with open(f"data/text/{fname}.txt", "w", encoding="utf-8") as t:
                t.write(text)
            
            idx += 1


if __name__ == "__main__":
    main()