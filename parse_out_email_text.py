#!/usr/bin/python

from nltk.stem.snowball import SnowballStemmer
import string

def parseOutText(f):
    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()

    ### split off metadata
    content = all_text.split("X-FileName:")
    words = ""
    if len(content) > 1:
        # removing punctuation
        exclude = set(string.punctuation)
        text_string = ''.join(ch for ch in content[1] if ch not in exclude)

        from nltk.tokenize import word_tokenize
        # tokenizing string into words
        words = word_tokenize(text_string)
        stemmer = SnowballStemmer('english')
        text = []
        # replacing words with their stems
        for word in words:
            text.append(stemmer.stem(word))
        # detokenizing the words
        text = "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in text]).strip()
        
    return text


