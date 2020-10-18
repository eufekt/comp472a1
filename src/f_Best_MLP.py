import numpy as np
import sklearn.neural_network
import sklearn.metrics         # for accuracy_score
from sklearn.model_selection import GridSearchCV

def best_mlp(train_csv, val_csv, test_with_label_csv, letter, prints=False):
    # split the matrices
    train_Y = train_csv[:, -1] 
    train_X = train_csv[:, :-1]

    # fit the data
    mlp = sklearn.neural_network.MLPClassifier()

    parameter_space = {
    'hidden_layer_sizes': [(30,50), (10,10,10)],
    'activation': ['logistic','tanh', 'relu', 'identity'],
    'solver': ['sgd', 'adam'],
    }

    clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
    clf.fit(train_X, train_Y)

    test_with_label_X = test_with_label_csv[:, :-1 ]
    test_with_label_Y = test_with_label_csv[:, -1]

    # prediction 
    test_with_label_Y_predict = clf.predict(test_with_label_X)

    # confusion matrix
    confusion_matrix = sklearn.metrics.confusion_matrix(test_with_label_Y, test_with_label_Y_predict) # confusion matrix

    # all the other stuff required
    report = sklearn.metrics.classification_report(test_with_label_Y, test_with_label_Y_predict, target_names=letter) 

    # classification array?
    index = np.arange(1, test_with_label_Y_predict.size + 1, 1) 
    array_to_write_to_file = np.stack((index, test_with_label_Y_predict), axis=1)

    if(prints):
        print('#####################')
        print('### best mpl ###')
        print('#####################')
        print('Best parameters found:\n', clf.best_params_)
        print(array_to_write_to_file)
        print(confusion_matrix)
        print(report)

    return  {
        'arr': array_to_write_to_file,
        'report': report,
        'confusionMatrix':confusion_matrix
    }   


