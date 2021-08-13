
import xgboost as xgb
import pandas as pd
import numpy as np
import sklearn
import itertools
import pickle
import psycopg2
import argparse
from sqlalchemy import create_engine
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description = 'The Main NLP process'
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

    #04.1 Parse the arguments
    args = parser.parse_args()

    #04.2 Connect to the SQLalchemy
    engine = create_engine('postgresql://postgres:{}@localhost:5432/hm_crdt'.format(args.in_password))

    #04.3 Extract Data
    Train_DF = pd.read_sql('''select * from abt.abt_train''', engine)#.sample(frac=1,random_state=198667)
    Test_Df = pd.read_sql('''select * from abt.abt_test''', engine)
    Sub_Df = pd.read_sql('''select * from abt.abt_kaggle_submission''', engine)

    #04.4 Load Model - XGBoost, top performing
    out_file = r'Final_Model_XGBoost.pkl'
    Model = pickle.load(open(out_file, "rb"))

    Features = Model.get_booster().feature_names

    #04.5 Data Prep
    y = Train_DF[['target','sk_id_curr']]
    X = Train_DF.drop('target', axis = 1)
    X = X.set_index('sk_id_curr')
    X = X[Features]
    y = y.set_index('sk_id_curr')

    y_test = Test_Df[['target','sk_id_curr']]
    X_test = Test_Df.drop('target', axis = 1)
    X_test = X_test.set_index('sk_id_curr')
    X_test = X_test[Features]
    y_test = y_test.set_index('sk_id_curr')

    y_sub = Sub_Df[['target','sk_id_curr']]
    X_sub = Sub_Df.drop('target', axis = 1)
    X_sub = X_sub.set_index('sk_id_curr')
    X_sub = X_sub[Features]
    y_sub = y_sub.set_index('sk_id_curr')


    #04.6 Normalization
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X) , columns=Features, index=X.index)
    X_test = pd.DataFrame(scaler.transform(X_test) , columns=Features, index=X_test.index)
    X_sub = pd.DataFrame(scaler.transform(X_sub) , columns=Features, index=X_sub.index)


    #04.7 Add Predictions and scores
    y['pd'] = Model.predict_proba(X)[:,1]
    y_test['pd'] = Model.predict_proba(X_test)[:,1]
    y_sub['pd'] = Model.predict_proba(X_sub)[:,1]
    y['predicted_default'] = Model.predict(X)
    y_test['predicted_default'] = Model.predict(X_test)
    y_sub['predicted_default'] = Model.predict(X_sub)


    #04.8 A function to convert probability to score
    def PD_2_Score(In_PD, Target_Score, PDO, T_Odds):
        '''
        converts a probability prediction to a score
        :param In_PD: The input Probability (of default) (float)
        :param Target_Score: The target score
        :param PDO: The points to double the odds
        :param T_Odds: The target odds
        :return: The output score (float)
        '''
        Odds = In_PD/(1-In_PD)
        factor = PDO/np.log(2)
        offset = Target_Score - factor*np.log(T_Odds)
        score = offset - factor*np.log(Odds)
        return(score)

    PD_Score_Trans = lambda x: PD_2_Score(x, 600, 50, 1)


    y['score'] = y['pd'].apply(PD_Score_Trans)
    y_test['score'] = y_test['pd'].apply(PD_Score_Trans)
    y_sub['score'] = y_sub['pd'].apply(PD_Score_Trans)


    #04.9 push to data warehouse
    y.to_sql('y_expl_train', engine, schema='misc', if_exists='replace')
    y_test.to_sql('y_expl_test', engine, schema='misc', if_exists='replace')
    y_sub.to_sql('y_expl_kagl', engine, schema='misc', if_exists='replace')

    
    #04.9 create table
    #Connect to DB
    conn = psycopg2.connect(
        "dbname='hm_crdt' user='{}' password='{}'".format(args.db_user, args.in_password)
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()

    feature_list_q = ','.join(Features)

    #04.10 combine datasets
    cur.execute(f"""create schema if not exists expl;
                
                
                create table expl.Base_Table as
                select ex.*
                    ,{feature_list_q}
                from abt.abt_train as bse
                inner join misc.y_expl_train as ex
                    on bse.sk_id_curr = ex.sk_id_curr
                    
                union
                
                select ex.*
                    ,{feature_list_q}
                from abt.abt_test as bse
                inner join misc.y_expl_test as ex
                    on bse.sk_id_curr = ex.sk_id_curr
                
                union
                
                select ex.sk_id_curr
                    ,null as target
                    ,ex.pd
                    ,ex.Predicted_Default
                    ,ex.Score
                    ,{feature_list_q}
                from abt.abt_kaggle_submission as bse
                inner join misc.y_expl_kagl as ex
                    on bse.sk_id_curr = ex.sk_id_curr   
                ;
                
                
                ALTER TABLE expl.Base_Table
                add primary key (SK_ID_CURR)
                ;
                
                """.format(feature_list_q))

    conn.close()
