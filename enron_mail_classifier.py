#!/usr/bin/python

import os.path
import pickle
import sys

sys.path.append( "../tools/" )
from parse_out_email_text import parseOutText

def main():

    from_sara  = open("from_sara.txt", "r")
    from_chris = open("from_chris.txt", "r")

    from_data = []
    word_data = []

    # we parse emails just once, then we save parsed data in pickled file
    if not os.path.exists("your_word_data.pkl"):
        for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
            for path in from_person:
                path = os.path.join('..', path[:-1])
                print(path)
                email = open(path, "r")
                text = parseOutText(email)
                # removing unwanted strings
                for string in ['sara', 'shackleton', 'chris', 'germani']:
                    if string in text:
                        text.replace(string, '')
                # word data will be our X
                word_data.append(text)

                # if mail is from sara, we append 0, else we append 1
                # from_data is our Y
                if name == 'sara':
                    from_data.append(0)
                else:
                    from_data.append(1)
                email.close()

        from_sara.close()
        from_chris.close()

        pickle.dump(word_data, open("your_word_data.pkl", "wb") )
        pickle.dump(from_data, open("your_email_authors.pkl", "wb") )
    else:
        print('< input file was found >\n')
        in_file = open("your_word_data.pkl", 'rb')
        word_data = pickle.load(in_file)
        out_file = open("your_email_authors.pkl", 'rb')
        from_data = pickle.load(out_file)
        in_file.close()
        out_file.close()

    # Vectorizing the data
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(word_data, from_data, test_size = 0.2)
    vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.5, stop_words = 'english')
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)

    # feature selection
    from sklearn.feature_selection import SelectPercentile, f_classif
    selection = SelectPercentile(f_classif, percentile=1)
    selection.fit(x_train, y_train)
    x_train = selection.transform(x_train).toarray()
    x_test = selection.transform(x_test).toarray()

    path = 'enron-classifier.pkl'
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    from sklearn.externals import joblib
    # saving the classifier for convenience
    if not os.path.exists(path):
        # training a classifier
        clf = clf.fit(x_train, y_train)
        joblib.dump(clf, path) # saving the classifier
    else:
        print('< classifier found >\n')
        clf = joblib.load(path) # if classifier is in a file, we extract
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    from sklearn.metrics import accuracy_score
    score = accuracy_score(pred, y_test)
    print('Accuracy score: ', round(score, 4))

if __name__ == '__main__':
    main()

