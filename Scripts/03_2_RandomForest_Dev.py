
import pandas as pd
import argparse
import numpy as np
import sklearn
import sklearn.ensemble
from sqlalchemy import create_engine
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description = 'Build the Random Forest'
    )

    parser.add_argument(
        'db_user',
        type=str,
        help='The name of the user'
    )

    parser.add_argument(
        'in_password',
        type=str,
        help='The password for the user'
    )

    #03_2.1 Parse the input arguments
    args = parser.parse_args()

    print('Creating Random Forest model')
    #03_2.2 Database engine
    engine = create_engine('postgresql://postgres:{}@localhost:5432/hm_crdt'.format(args.in_password))

    #03_2.3 Extract from DB
    Train_DF = pd.read_sql('''select * from abt.abt_train''', engine)#.sample(frac=1,random_state=198667)
    Test_Df = pd.read_sql('''select * from abt.abt_test''', engine)
    Sub_Df = pd.read_sql('''select * from abt.abt_kaggle_submission''', engine)

    Train_DF = Train_DF.drop(['sk_id_curr'], axis = 1)
    Test_Df = Test_Df.drop(['sk_id_curr'], axis = 1)
    Sub_Df = Sub_Df.set_index('sk_id_curr')
    Sub_Df = Sub_Df.drop(['target'], axis = 1)

    X = Train_DF.drop('target', axis = 1)
    y = Train_DF['target']

    X_test = Test_Df.drop('target', axis = 1)
    y_test = Test_Df['target']

    #03_2.4 The evaluation function. Same in every function
    def Evaluation(Y_True, Y_Predict, Y_Predict_prob):
        Output = {}
        Output['Accuracy'] = sklearn.metrics.accuracy_score(Y_True, Y_Predict)
        Output['Precision'] = sklearn.metrics.precision_score(Y_True, Y_Predict)
        Output['Recall'] = sklearn.metrics.recall_score(Y_True, Y_Predict)
        Output['AUC'] = sklearn.metrics.roc_auc_score(Y_True, Y_Predict_prob)
        return(Output)


    #Remove correlated varaiables
    threshold = 0.85

    # Absolute value correlation matrix
    corr_matrix = X.corr().abs()
    corr_matrix.head()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    upper.head()
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    X = X.drop(to_drop, axis = 1)
    X_test = X_test.drop(to_drop, axis = 1)
    Sub_Df = Sub_Df.drop(to_drop, axis = 1)

    #Feature Scaling
    Colnames = X.columns
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X) , columns=Colnames)
    X_test = pd.DataFrame(scaler.transform(X_test) , columns=Colnames)
    Sub_Df = pd.DataFrame(scaler.transform(Sub_Df) , columns=Colnames, index=Sub_Df.index)


    Feature_set_iter = X.columns


    sel_params = {
            'n_estimators': 200,
            'class_weight':'balanced',
            'min_impurity_decrease':10**(-3.5),
            #'max_features':0.1,
            'random_state':198666,
            'verbose':2,
            'n_jobs':-2
            }

    y_train_sel, y_CV_sel, X_train_sel, X_CV_sel = train_test_split(y, X, test_size=0.2, random_state=198666, stratify = y)

    RF_sel = sklearn.ensemble.RandomForestClassifier(**sel_params)
    model_sel = RF_sel.fit(X_train_sel, y_train_sel)



    y_sel_train_Predict = model_sel.predict(X_train_sel)
    y_sel_train_Predict_prob = model_sel.predict_proba(X_train_sel)[:,1]
    y_sel_CV_Predict = model_sel.predict(X_CV_sel)
    y_sel_CV_Predict_prob = model_sel.predict_proba(X_CV_sel)[:,1]
    print(Evaluation(y_train_sel, y_sel_train_Predict, y_sel_train_Predict_prob))
    print(Evaluation(y_CV_sel, y_sel_CV_Predict, y_sel_CV_Predict_prob))


    #Feature selection
    feature_imp = model_sel.feature_importances_
    feature_imp_df = pd.DataFrame({'feature':Feature_set_iter, 'importances':feature_imp})



    Feature_sel = feature_imp_df['feature'][feature_imp_df['importances'] > np.percentile(feature_imp_df['importances'],90)]

    X = X[Feature_sel]
    X_test = X_test[Feature_sel]
    Sub_Df = Sub_Df[Feature_sel]


    y_train, y_CV, X_train, X_CV = train_test_split(y, X, test_size=0.2, random_state=198666, stratify = y)

    params = sel_params.copy()
    params['n_estimators'] = 300
    RF = sklearn.ensemble.RandomForestClassifier(**params)
    Model = RF.fit(X_train, y_train)

    y_train_pred = Model.predict(X_train)
    y_CV_pred = Model.predict(X_CV)
    y_train_pred_prob = Model.predict_proba(X_train)[:,1]
    y_CV_pred_prob = Model.predict_proba(X_CV)[:,1]

    print(Evaluation(y_train, y_train_pred, y_train_pred_prob))
    print(Evaluation(y_CV, y_CV_pred, y_CV_pred_prob))


    kf = KFold(n_splits=5, shuffle=True, random_state=198667)

    CV_params = params.copy()
    CV_params['verbose'] = 0

    for fold, (train_index, CV_index) in enumerate(kf.split(X), 1):
        X_train_kf = X.iloc[train_index]
        y_train_kf = y.iloc[train_index]  # Based on your code, you might need a ravel call here, but I would look into how you're generating your y
        X_CV_kf = X.iloc[CV_index]
        y_CV_kf = y.iloc[CV_index]  # See comment on ravel and  y_train

        model_kf = sklearn.ensemble.RandomForestClassifier(**CV_params)
        model_kf.fit(X_train_kf, y_train_kf)
        y_CV_pred_kf = model_kf.predict(X_CV_kf)
        y_train_pred_kf = model_kf.predict(X_train_kf)
        y_CV_pred_prob_kf = model_kf.predict_proba(X_CV_kf)[:,1]
        y_train_pred_prob_kf = model_kf.predict_proba(X_train_kf)[:,1]
        print(f'Fold {fold}')
        print(f'Train Stats')
        print(Evaluation(y_train_kf,y_train_pred_kf,y_train_pred_prob_kf))
        print(f'Test Stats')
        print(Evaluation(y_CV_kf,y_CV_pred_kf,y_CV_pred_prob_kf))


    #Evaluation
    y_test_pred = Model.predict(X_test)
    y_test_pred_prob = Model.predict_proba(X_test)[:,1]
    print('Evaluation for test set')
    print(Evaluation(y_test,y_test_pred,y_test_pred_prob))

    #Kaggle Submission
    Kagl_Sub = Sub_Df.copy()
    Kagl_Sub['TARGET'] = 0.5
    Kagl_Sub['TARGET'] = Model.predict_proba(Sub_Df)[:,1]
    Kagl_Sub = Kagl_Sub['TARGET'].copy()
    Kagl_Sub.to_csv('Kaggle_Submissions/02_Random_Forest.csv')
