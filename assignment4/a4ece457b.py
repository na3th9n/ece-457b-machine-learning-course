"""
- scikit-learn==1.7.2
- scipy==1.16.3
- numpy==2.3.5
"""

from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import accuracy_score
from scipy.sparse import vstack
import numpy as np

def main():
    dtrain = datasets.fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'))
    dtest = datasets.fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'))

    # training data
    X_train_raw_all = dtrain.data
    y_train_all = dtrain.target

    # test data
    X_test_raw = dtest.data
    y_test = dtest.target

    # create a training set and a validation set for hyperparameter tuning
    X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(
        X_train_raw_all,
        y_train_all,
        test_size=0.2,
        random_state=1,
        stratify=y_train_all
    )

    tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        max_df=0.5,      
        min_df=5,      
        ngram_range=(1, 2)  
    )

    X_tr = tfidf.fit_transform(X_tr_raw)
    X_val = tfidf.transform(X_val_raw)
    X_test = tfidf.transform(X_test_raw)

    X_full = vstack([X_tr, X_val])
    y_full = np.concatenate([y_tr, y_val])

    C_values = [0.01, 0.1, 1.0, 10.0]

    def tune_and_eval(name, make_model):
        best_C = None
        best_val_acc = 0.0

        for C in C_values:
            clf = make_model(C)
            clf.fit(X_tr, y_tr)
            y_val_pred = clf.predict(X_val)
            acc = accuracy_score(y_val, y_val_pred)

            if acc > best_val_acc:
                best_val_acc = acc
                best_C = C

        clf_final = make_model(best_C)
        clf_final.fit(X_full, y_full)

        y_test_pred = clf_final.predict(X_test)
        test_acc = accuracy_score(y_test, y_test_pred)

        return clf_final, best_C, best_val_acc, test_acc

    def make_linear_svc(C):
        return LinearSVC(C=C, random_state=0)

    def make_ovr(C):
        base = LinearSVC(C=C, random_state=0)
        return OneVsRestClassifier(base)

    def make_ovo(C):
        base = LinearSVC(C=C, random_state=0)
        return OneVsOneClassifier(base)

    clf_plain, C_plain, val_plain, test_plain = tune_and_eval("LinearSVC (internal OvR)", make_linear_svc)

    clf_ovr, C_ovr, val_ovr, test_ovr = tune_and_eval("OvR(LinearSVC)", make_ovr)

    clf_ovo, C_ovo, val_ovo, test_ovo = tune_and_eval("OvO(LinearSVC)", make_ovo)

    print("FINAL TEST ACCURACIES")
    print(f"SVM            : C={C_plain},   test acc={test_plain:.4f}")
    print(f"OvR(LinearSVC) : C={C_ovr},   test acc={test_ovr:.4f}")
    print(f"OvO(LinearSVC) : C={C_ovo},   test acc={test_ovo:.4f}")

if __name__ == "__main__":
    main()
