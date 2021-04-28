import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


if __name__ == '__main__':

    data_set = 'SAKI Exercise 1 - Transaction Classification - Data Set.csv'
    df = pd.read_csv(data_set, sep=';', index_col=0, na_values='?')

    #################################################################################
    # 1. clean and prepare data                                                     #
    #################################################################################

    df['Buchungstext'] = df['Buchungstext'].str.replace(r'\W', ' ', regex=True)
    df['Buchungstext'] = df['Buchungstext'].str.lower()
    df['Verwendungszweck'] = df['Verwendungszweck'].str.replace(r'\W', ' ', regex=True)
    df['Verwendungszweck'] = df['Verwendungszweck'].str.replace(r'\d', '', regex=True)
    df['Verwendungszweck'] = df['Verwendungszweck'].str.replace(r'\b\w{1,2}\b', '', regex=True)
    df['Verwendungszweck'] = df['Verwendungszweck'].str.lower()
    df['Beguenstigter/Zahlungspflichtiger'] = df['Beguenstigter/Zahlungspflichtiger'].str.replace(r'\W', ' ', regex=True)
    df['Beguenstigter/Zahlungspflichtiger'] = df['Beguenstigter/Zahlungspflichtiger'].str.replace(r'\d', '', regex=True)
    df['Beguenstigter/Zahlungspflichtiger'] = df['Beguenstigter/Zahlungspflichtiger'].str.lower()

    #################################################################################
    # 2. label data                                                                 #
    #################################################################################

    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])

    #################################################################################
    # 3. define features to use                                                     #
    #################################################################################

    data = pd.DataFrame({'message': []})
    data['message'] = df['Buchungstext'] + ' ' + df['Verwendungszweck'] + ' ' + df['Beguenstigter/Zahlungspflichtiger']

    #################################################################################
    # 4. transform features into a usable format and 5. train model                 #
    #################################################################################

    X_train, X_test, y_train, y_test = train_test_split(data['message'], df['label'])

    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('clf', MultinomialNB()),
    ])
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
    }
    gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)
    gs_clf.fit(X_train, y_train)

    #################################################################################
    # 6. evaluate model                                                             #
    #################################################################################

    predictions = gs_clf.predict(X_test)
    print(classification_report(y_test, predictions, target_names=le.classes_))

    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))


