from sklearn.model_selection import GridSearchCV
import sklearn.tree
import numpy as np
import sklearn.metrics         # for accuracy_score
import sys


def best_DT(train_csv, val_csv, test_with_label_csv, letter, prints=False):

    param_values = {
        'criterion':['gini', 'entropy'], 
        'max_depth': [10, None], 
        'min_samples_split':[2, 3, 4, 5, 6, 7, 8, 9 ,10, 11, 15, 20, 25, 30, 40, 50], 
        'min_impurity_decrease':[0.0, 0.1,0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9, 1,2,3], 
        'class_weight': [None, 'balanced'], 
    }

    # setup the model
    # base_dt = GridSearchCV(sklearn.tree.DecisionTreeClassifier(), param_values)
    best_dt = sklearn.tree.DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_weight_fraction_leaf=0.0, min_impurity_decrease=0.0, class_weight=None)

    CV_base_dt = GridSearchCV(estimator=best_dt, param_grid=param_values, cv = 3)

    # print(CV_base_dt.best_score_)

    #setup the training set 
    train_Y = train_csv[:, -1] 
    train_X = train_csv[:, :-1]

    # train the model
    CV_base_dt.fit(train_X, train_Y)



    # set up the testing data
    test_with_label_X = test_with_label_csv[:, :-1 ]
    test_with_label_Y = test_with_label_csv[:, -1]

    # prediction 
    test_with_label_Y_predict = CV_base_dt.predict(test_with_label_X)

    # cinfusion matrix
    confusion_matrix = sklearn.metrics.confusion_matrix(test_with_label_Y, test_with_label_Y_predict)

     # report
    report = sklearn.metrics.classification_report(test_with_label_Y, test_with_label_Y_predict, target_names=letter) 

    index = np.arange(1, test_with_label_Y_predict.size + 1, 1) 
    array_to_write_to_file = np.stack((index, test_with_label_Y_predict), axis=1)

    if(prints):
        print('#####################')
        print('### best dt ###')
        print('#####################')
        print("params\n")
        print(CV_base_dt.best_params_)
        print("\nbest score\n")
        print(CV_base_dt.best_score_)
        print()
        print(array_to_write_to_file)
        print(confusion_matrix)
        print(report)

    return  {
        'arr': array_to_write_to_file,
        'report': report,
        'confusionMatrix':confusion_matrix
        }
