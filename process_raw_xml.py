"""
This script processes the raw .xml files in the canonical greek lit repo, 
purges them of their xml scaffolding, and stores the text body as .txt 
files.
Note on the source data: We have used data from the following repos: 
https://github.com/OpenGreekAndLatin/First1KGreek/
https://github.com/PerseusDL/canonical-greekLit/
"""
import os
import re
import random
random.seed(112310)
from glob import glob
from bs4 import BeautifulSoup
from collections import defaultdict
from tabulate import tabulate
import nltk
nltk.download('punkt_tab')
import tqdm


def process_raw_xml():
    # Store xml data as clean raw text

    # Exclusion list of english words that appear in the by-line of the Ancient Greek texts
    exclude = [
        "This pointer", 
        "Keyboarding", 
        "professional data entry", 
        "The following",
        "optical",
        "Notes",
        "encoded",
        "copyright",
        "Lace"
    ]

    authors = []

    # Create subfolder if it does not exist
    if not os.path.exists("data/text"):
        os.makedirs("data/text")

    # Iterate over all the relevant .xml files, and extract the raw text, 
    # the title and the author, and store each as a text file
    idx = 0
    for f in glob("data/xml/**/*grc*.xml", recursive=True):
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

            fname = f"{author_tags[0].text}{idx}"
            fname = "".join(c for c in fname if c.isalnum())
            text = f"{title_tags[0].text} by {author_tags[0].text}" + " \n " \
                + " \n ".join(el.text for el in text_tags if not any(w in el.text for w in exclude))
            
            with open(f"data/text/{fname}.txt", "w", encoding="utf-8") as t:
                t.write(text)
            
            idx += 1


def print_dataset_summary(limit=50):
    # Start with number of documents and number of documents per author
    docs_per_author = defaultdict(int)
    lines_per_author = defaultdict(int)

    for file in tqdm.tqdm(os.listdir("data/text")):
        author = "".join([c for c in file if not c.isdigit()]).split(".")[0]
        docs_per_author[author] += 1

        with open(f"data/text/{file}", "r") as f:
            g = f.read()
            lines_per_author[author] += len(nltk.sent_tokenize(g))
    
    summary = tabulate(
        [
            [author, docs_per_author[author], lines_per_author[author]]
            for author in sorted(docs_per_author, key=docs_per_author.get, reverse=True)
        ][:limit],
        headers=["Author", "Number of documents", "Number of lines of text"]
    )
    print(summary)


def prepare_data_for_authorship_classification():
    # Following the instructions outlined in Yamshchikov et al.'s paper,
    # we will limit the dataset to works by the following authors:

    #     Galenus, Origenes, Plutarch, Cassius Dio, Flavius Josephus, 
    #     Philo Judaeus, Athenaeus, Claudius Ptolemaeus, Aelius Aristides, 
    #     Strabo, Lucianus, Clemens Alexandrinus, Appianus, Pausanias, 
    #     Sextus Empiricus, Dio Chrysostomus.

    # Furthermore, we will randomly n sentences, where n is the smallest
    # number of sentences attributable to any of the above authors, to 
    # avoid bias.

    selected_authors = [
        "Galen", "Origenes", "Plutarch", "CassiusDio", "FlaviusJosephus",
        "PhiloJudaeus", "Athenaeus", "Claudius", "AeliusAristides", 
        "Strabo", "Lucian", "ClementofAlexandria", "Appian", "Pausanias",
        "Sextus", "DioChrysostom"
    ]

    sentences_by_author = {}
    for file in tqdm.tqdm(os.listdir("data/text")):
        author = "".join([c for c in file if not c.isdigit()]).split(".")[0]
        for idx, selected_author in enumerate(selected_authors):
            if selected_author in author:
                with open(f"data/text/{file}", "r") as f:
                    g = f.read().replace("\n", " ").replace("\r", "").replace("\t", " ")
                if idx in sentences_by_author:
                    sentences_by_author[idx].extend(nltk.sent_tokenize(g)[1:])
                else:
                    sentences_by_author[idx] = nltk.sent_tokenize(g)[1:]

    if not os.path.exists("data/authorship_classification"):
        os.makedirs("data/authorship_classification")

    # Find the minimum number of sentences
    min_sentence_cnt = float("inf")
    for author_idx in sentences_by_author:
        if len(sentences_by_author[author_idx]) < min_sentence_cnt:
            min_sentence_cnt = len(sentences_by_author[author_idx])

    # Randomly sample `min_sentence_cnt` number of sentences per author
    for author_idx in sentences_by_author:
        sentences_by_author[author_idx] = random.sample(sentences_by_author[author_idx], min_sentence_cnt)

    # Randomly shuffle into train (80%), dev (10%) and test (10%) sets, and store dataset
    # Randomly sample `min_sentence_cnt` number of sentences per author

    with open(f"data/authorship_classification/train.txt", "a") as f:
        f.write(f"sentence_id \t sentence \t author_id \n")
    with open(f"data/authorship_classification/dev.txt", "a") as f:
        f.write(f"sentence_id \t sentence \t author_id \n")
    with open(f"data/authorship_classification/test.txt", "a") as f:
        f.write(f"sentence_id \t sentence \t author_id \n")

    cnt_80_pct = int(min_sentence_cnt * 0.8)
    cnt_10_pct = int(min_sentence_cnt * 0.1)

    for author_idx in tqdm.tqdm(sentences_by_author):

        random.shuffle(sentences_by_author[author_idx])
        train = sentences_by_author[author_idx][:cnt_80_pct]
        dev = sentences_by_author[author_idx][cnt_80_pct:cnt_80_pct+cnt_10_pct]
        test = sentences_by_author[author_idx][cnt_80_pct+cnt_10_pct:cnt_80_pct+cnt_10_pct+cnt_10_pct]
        
        with open(f"data/authorship_classification/train.txt", "a") as f:
            for idx, sentence in enumerate(train):
                f.write(f"{idx} \t {sentence} \t {author_idx} \n")
        with open(f"data/authorship_classification/dev.txt", "a") as f:
            for idx, sentence in enumerate(dev):
                f.write(f"{idx} \t {sentence} \t {author_idx} \n")
        with open(f"data/authorship_classification/test.txt", "a") as f:
            for idx, sentence in enumerate(test):
                f.write(f"{idx} \t {sentence} \t {author_idx} \n")        
    

if __name__ == "__main__":
    # process_raw_xml()
    # print_dataset_summary()
    prepare_data_for_authorship_classification()