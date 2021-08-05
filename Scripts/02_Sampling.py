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

    args = parser.parse_args()

    #Connect to DB
    conn = psycopg2.connect(
        "dbname='hm_crdt' user='{}' password='{}'".format(args.db_user, args.in_password)
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()

    engine = create_engine('postgresql://postgres:{}@localhost:5432/hm_crdt'.format(args.in_password))

    print('Extracting from DB')
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
    Full_DF.columns = [i.strip().lower().replace(' ','_').replace('/','_').replace(',','_').replace('+','') for i in col_list]


    Full_DF_kagl = Full_DF[pd.isna(Full_DF['target'])]
    Train_Test = Full_DF[~pd.isna(Full_DF['target'])]
    Train_Test['target'] = Train_Test['target'].astype(int).copy()

    Train , Test = train_test_split(Train_Test, test_size=0.2, random_state=198666, stratify = Train_Test['target'])


    #Check if Strafied correctly
    sum(Train['target'])/Train['target'].count()
    sum(Test['target'])/Test['target'].count()


    #Outlier Detection
    OD_Model = IsolationForest(random_state=198666).fit(Train.drop('target',axis=1))
    Outlier_array = OD_Model.predict(Train.drop('target',axis=1))
    Train = Train[Outlier_array == 1].copy()


    #Oversampling
    #oversample = SMOTE()
    #Train_SMOTE, Train_SMOTE_Target = oversample.fit_resample(Train.drop(['target'],axis=1), Train['target'])
    #Train_SMOTE = pd.DataFrame(Train_SMOTE_Target).merge(Train_SMOTE, left_index=True, right_index=True)


    #Check if SMOTE was successful
    #sum(Train_SMOTE['target'])/Train_SMOTE['target'].count()


    print('Pushing Kaggle submission dataset to DB')
    Full_DF_kagl.to_sql('abt_kaggle_submission', engine, schema='abt', if_exists='replace', chunksize=10000)
    print('Pushing Train dataset to DB')
    Train.to_sql('abt_train', engine, schema='abt', if_exists='replace', chunksize=10000)
    print('Pushing Test dataset to DB')
    Test.to_sql('abt_test', engine, schema='abt', if_exists='replace', chunksize=10000)
    #Train_SMOTE.to_sql('abt_train_smote', engine, schema='abt', if_exists='replace')



    conn.close()
