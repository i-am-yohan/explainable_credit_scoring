import psycopg2
import argparse
import sys
import pandas as pd
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import pandas.io.sql as psql

if __name__ == '__main__':

    #2.1 The arguments
    parser = argparse.ArgumentParser(
        description = 'The train and test table creation'
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

    args = parser.parse_args()

    #2.2 Connect to DB
    conn = psycopg2.connect(
        "dbname='hm_crdt' user='{}' password='{}'".format(args.db_user, args.in_password)
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()

    #2.3 Connect to SQLalchemy engine
    engine = create_engine('postgresql://postgres:{}@localhost:5432/hm_crdt'.format(args.in_password))

    print('Extracting from DB')
    #2.4 extract full table
    #2.4.1 must do it in chunks to save memory
    Full_DF_chunk = pd.read_sql('''
                    select *
                    from abt.abt_full
                    ;'''
                    , con=conn
                    , chunksize=10000
                    )

    Full_DF = pd.DataFrame([])
    for chunk in Full_DF_chunk:
        Full_DF = Full_DF.append(chunk)


    Full_DF = Full_DF.set_index(['sk_id_curr']).copy()
    Full_DF = pd.get_dummies(Full_DF)

    col_list = list(Full_DF.columns)
    #2.4.2 change all columns to lower case. postgres only likes lower case
    Full_DF.columns = [i.strip().lower().replace(' ','_').replace('/','_').replace(',','_').replace('+','') for i in col_list]


    #2.5 Train and test creation
    Full_DF_kagl = Full_DF[pd.isna(Full_DF['target'])] #These are cases from the kaggle test test, they have no target labels
    Train_Test = Full_DF[~pd.isna(Full_DF['target'])] 
    Train_Test['target'] = Train_Test['target'].astype(int).copy()

    #2.5.1 The train and test split
    Train , Test = train_test_split(Train_Test, test_size=0.2, random_state=198666, stratify = Train_Test['target'])


    #2.5.2 Check if Strafied correctly
    sum(Train['target'])/Train['target'].count()
    sum(Test['target'])/Test['target'].count()


    #2.6 Outlier Detection and removal
    OD_Model = IsolationForest(random_state=198666).fit(Train.drop('target',axis=1))
    Outlier_array = OD_Model.predict(Train.drop('target',axis=1))
    Train = Train[Outlier_array == 1].copy()


    #2.7 Push to database
    print('Pushing Kaggle submission dataset to DB')
    Full_DF_kagl.to_sql('abt_kaggle_submission', engine, schema='abt', if_exists='replace', chunksize=10000)
    print('Pushing Train dataset to DB')
    Train.to_sql('abt_train', engine, schema='abt', if_exists='replace', chunksize=10000)
    print('Pushing Test dataset to DB')
    Test.to_sql('abt_test', engine, schema='abt', if_exists='replace', chunksize=10000)
    #Train_SMOTE.to_sql('abt_train_smote', engine, schema='abt', if_exists='replace')

    conn.close()
