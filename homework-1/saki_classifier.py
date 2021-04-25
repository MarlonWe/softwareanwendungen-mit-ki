import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

if __name__ == '__main__':

    # 1. clean and prepare data
    data_set = 'SAKI Exercise 1 - Transaction Classification - Data Set.csv'
    df = pd.read_csv(data_set, sep=';', index_col=0, na_values='?')

    df['Buchungstext'] = df['Buchungstext'].str.replace(r'\W', ' ', regex=True)
    df['Buchungstext'] = df['Buchungstext'].str.lower()
    df['Verwendungszweck'] = df['Verwendungszweck'].str.replace(r'\W', ' ', regex=True)
    df['Verwendungszweck'] = df['Verwendungszweck'].str.replace(r'\d', '', regex=True)
    df['Verwendungszweck'] = df['Verwendungszweck'].str.replace(r'\b\w{1,2}\b', '', regex=True)
    df['Verwendungszweck'] = df['Verwendungszweck'].str.lower()
    df['Beguenstigter/Zahlungspflichtiger'] = df['Beguenstigter/Zahlungspflichtiger'].str.replace(r'\W', ' ', regex=True)
    df['Beguenstigter/Zahlungspflichtiger'] = df['Beguenstigter/Zahlungspflichtiger'].str.replace(r'\d', '', regex=True)
    df['Beguenstigter/Zahlungspflichtiger'] = df['Beguenstigter/Zahlungspflichtiger'].str.lower()

    print(df[df['label'] == "living"])
    # 2. label data
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])

    data = pd.DataFrame({'message': []})
    data['message'] = df['Buchungstext'] + ' ' + df['Verwendungszweck'] + ' ' + df['Beguenstigter/Zahlungspflichtiger']

    # 3. define features to use
    X_train, X_test, y_train, y_test = train_test_split(data['message'], df['label'])

    # 4. transform features into a usable format and 5. train model
    text_clf = Pipeline([
        ('vect', TfidfVectorizer(sublinear_tf=True, max_df=0.5)),
        ('clf', ComplementNB()),
    ])
    parameters = {
        'vect__ngram_range': [(1, 2), (1, 3), (1, 4), (2, 2)],
        'clf__alpha': np.linspace(0.1, 1, 5),
        'clf__norm': [True, False]
    }

    gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)
    gs_clf.fit(X_train, y_train)

    # 5. evaluate model
    predictions = gs_clf.predict(X_test)
    print(classification_report(y_test, predictions, target_names=le.classes_))

    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

    #scores = cross_val_score(gs_clf.estimator, X_train, y_train, cv=5)
    # 7. visualize results


