import xgboost as xgb
import pandas as pd
import numpy as np
import pickle
import math
import itertools
import sys
import progressbar
import psycopg2
import argparse
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# WARNING!! There's a lot of redundant code here which I didnt have time to remove. Does not have a huge effect on performance
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='The program that extracts and uploads the counterfactuals'
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

    parser.add_argument(
        'acc_id',
        type=int,
        help='The account id (sk_id_curr) to investigate'
    )

    parser.add_argument(
        'target_score',
        type=int,
        help='The target credit score'
    )

    parser.add_argument(
        'lower_bound',
        type=int,
        help='The lower bound credit score'
    )

    parser.add_argument(
        'upper_bound',
        type=int,
        help='The upper bound credit score'
    )

    parser.add_argument(
        'n_samples',
        type=int,
        help='The number of samples to take at each iteration of the algo'
    )

    args = parser.parse_args()

    engine = create_engine('postgresql://postgres:{}@localhost:5432/hm_crdt'.format(args.in_password))

    # 05.1 Extract from Database
    Expl_DF = pd.read_sql('''select * from expl.base_table''', engine)

    # 05.2 Load in XGBoost model
    out_file = r'Final_Model_XGBoost.pkl'
    Model = pickle.load(open(out_file, "rb"))

    Features = Model.get_booster().feature_names

    # 05.2 Get feature importances
    feature_imp = Model.feature_importances_
    feature_imp_arr = np.array(
        Model.feature_importances_)  # np.reshape(np.array(Model.feature_importances_) , newshape = [len(feature_imp),1])
    feature_imp_df = pd.DataFrame({'feature': Features, 'importances': feature_imp})

    feature_imp_df = feature_imp_df.sort_values(by=['importances'], ascending=False)

    # 05.3 Normalize features
    scaler = StandardScaler()  # = MinMaxScaler()
    Expl_DF_norm = Expl_DF.copy()
    Expl_DF_norm[Features] = pd.DataFrame(scaler.fit_transform(Expl_DF_norm[Features]), columns=Features)

    # 05.3 Get Initial Data and perform the ususal operations....
    Train_DF = pd.read_sql('''select * from abt.abt_train''', engine)  # .sample(frac=1,random_state=198667)
    Test_Df = pd.read_sql('''select * from abt.abt_test''', engine)
    Sub_Df = pd.read_sql('''select * from abt.abt_kaggle_submission''', engine)

    Train_DF = Train_DF.set_index('sk_id_curr')
    Test_Df = Test_Df.set_index('sk_id_curr')
    Sub_Df = Sub_Df.set_index('sk_id_curr')
    Sub_Df = Sub_Df.drop(['target'], axis=1)

    X = Train_DF[Features]
    y = Train_DF['target']

    X_test = Test_Df[Features]
    y_test = Test_Df['target']

    X_Kagl = Sub_Df[Features]

    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=Features, index=X.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=Features, index=X_test.index)
    X_Kagl = pd.DataFrame(scaler.transform(X_Kagl), columns=Features, index=X_Kagl.index)

    X_Full = X.append(X_test).append(X_Kagl)


    def PD_2_Score(In_PD, Target_Score, PDO, T_Odds):
        Odds = In_PD / (1 - In_PD)
        factor = PDO / np.log(2)
        offset = Target_Score - factor * np.log(T_Odds)
        score = offset - factor * np.log(Odds)
        return (score)


    # 05.4 A loading bar for the algorithm
    # Copied and pasted from stack OVerflow
    # https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
        # Print New Line on Complete
        if iteration == total:
            print()


    # 05.5 A distance function used when finding counterfactuals
    def setdiff_nd_positivenums(a, b):
        s = np.maximum(a.max(0) + 1, b.max(0) + 1)
        return a[~np.isin(a.dot(s), b.dot(s))]


    # 05.6 This is the BIG function. Where the entire counterfactual extraction is completed.
    # Extract and find counterfactuals
    def Explain(Search_ID, Target_Score, Lower_Bound, Upper_Bound, N_samples):
        '''
        :param Search_ID: the sk_id_curr for which requires explanation
        :param Target_Score: the ideal score to target
        :param Lower_Bound: the lower bound score
        :param Upper_Bound: the upper bound score
        :param N_samples: the number of samples to take at each generation
        :return: The output pandas table to be pushed to the DB
        '''
        # global Gentic_df

        # 05.6.1 Create empty dataframe to store all analysis results
        # Gentic_df is the main table!!!
        Gentic_df = pd.DataFrame(columns=Features, dtype=bool)
        Gentic_df['Generation'] = None
        Gentic_df['Feature_id'] = None
        Gentic_df['Score'] = None
        Gentic_df['sk_id_curr'] = None
        Gentic_df['Score'] = Gentic_df['Score'].astype(float).copy()
        Gentic_df['Feature_id'] = Gentic_df['Feature_id'].astype(str).copy()
        Gentic_df['sk_id_curr'] = Gentic_df['sk_id_curr'].astype(str).copy()

        # 05.6.2 Create empty boolean dataframe
        pre_0_df = {}
        for feat in Features:
            pre_0_df[feat] = False

        # 05.7 This deploys the NUN algorithm and makes the feature value changes to the case that requires explanation
        def CF_Score_Calc(Search_ID, Lower_Bound_i, Upper_Bound_i, feat_i):
            '''
            Deploy NUN algo and apply counterfactual changes
            :param Search_ID: the sk_id_curr for which requires explanation
            :param Lower_Bound_i: the lower bound score
            :param Upper_Bound_i: the upper bound score
            :param feat_i: the features to exclude when looking for NUN
            :return: A dataframe for the process to work with
            '''

            for feat_i_i in feat_i:  # Change all features in boolean DF
                pre_0_df[feat_i_i] = False
            Gentic_df_i = pd.DataFrame(pre_0_df, index=[
                0])  # Create iteration of Gentic_df that will be appended to the main table
            Gentic_df_i['Generation'] = 1
            Gentic_df_i[
                feat_i] = True  # Change the features to exclude in NUN in Gentic DF to true. Used to identify the features
            CF_Test_DF = Find_CF(Search_ID, Lower_Bound_i, Upper_Bound_i, feat_i)  # Deploy the NUN algo
            CF_Feat_vec = X_Full.loc[[CF_Test_DF.loc[
                                          'Counterfactual', 'sk_id_curr']]].copy()  # Extract the row from X_Full for the counterfactual
            Test_Feat_vec = X_Full.loc[
                [Search_ID]].copy()  # Extract the row from X_Full for the case that requires explanation
            Test_Feat_vec[feat_i] = CF_Feat_vec[
                feat_i].values  # Important!! Replace the values in the case with those of the counterfactual
            Prdict = Model.predict_proba(X_Full.loc[[Search_ID]])[:, 1]
            New_PD = Model.predict_proba(Test_Feat_vec)[:,
                     1]  # Make a prediction using the case with the new counterfactual values
            New_Score = PD_2_Score(New_PD, 600, 50, 1)  # Convert the probability predictions to a score value
            Gentic_df_i['Score'] = New_Score  # Add the new score value to table
            Gentic_df_i['sk_id_curr'] = CF_Test_DF.loc['Counterfactual', 'sk_id_curr']
            return (Gentic_df_i)

        # 05.8 Finds the nearest score within each tolerance
        def Find_CF(Search_ID, Lower_Bound_j, Upper_Bound_j, Feat_Chck=[]):  # , Tolerance):
            '''
            Find the NUN while excluding a set of features
            :param Search_ID: the case that requires explanation
            :param Lower_Bound_j: The lower search limit
            :param Upper_Bound_j: The upper search limit
            :param Feat_Chck: the features to excluded when searching for the NUN
            :return:
            '''
            Feat_Chck_plus = Feat_Chck + ['sk_id_curr', 'target', 'pd', 'predicted_default',
                                          'score']  # Create a vector of features to exclude.
            Init_Row = Expl_DF[Expl_DF['sk_id_curr'] == Search_ID]  # Extract the case from the Explanation DF
            Init_Row.index = ['Case']
            Init_Row_norm = Expl_DF_norm[Expl_DF_norm[
                                             'sk_id_curr'] == Search_ID]  # Extract the normalized features for the case that requires explanation
            Init_arr_norm = np.array(
                Init_Row_norm.drop(Feat_Chck_plus, axis=1))  # remove features that are to be excluded from NUN
            Score_Dim = (Expl_DF[['score']] >= Lower_Bound_j) & (Expl_DF[[
                'score']] <= Upper_Bound_j)  # Create a boolean vector to subset the DF to only the scores within the required threshold
            CF_df = Expl_DF.loc[Score_Dim.values,].copy()  # Apply subsetting
            CF_df_norm = Expl_DF_norm.loc[Score_Dim.values,].copy()  # Extract the normalized values
            CF_arr_norm = np.array(CF_df_norm.drop(Feat_Chck_plus, axis=1))  # Drop the excluded feature columns
            Distance = np.sum(np.abs(CF_arr_norm - Init_arr_norm),
                              axis=1)  # Calculate the normalized uclidian distance between the case and all of the counterfactuals
            CF_df['Distance'] = Distance
            CF = CF_df[CF_df['Distance'] == min(CF_df['Distance'])].drop('Distance',
                                                                         axis=1)  # Find the observations with the smallest distance
            CF.index = ['Counterfactual']
            Output = Init_Row.append(CF)  # Append
            return (Output)

        # 05.9 The output part....
        def Out():
            '''
            :return: This returns the output dataframe and terminates the program
            '''
            sys.stdout.write('Outputting Counterfactuals')
            # Explain_out = {}
            Gentic_df_out = Gentic_df.loc[(Gentic_df['Score'] >= Lower_Bound) & (Gentic_df['Score'] <= Upper_Bound),
                            :]  # Extract cases in main DF that satisfy the criteria
            # Explain_out['Binary_Feature_Table'] = Gentic_df_out

            for i in range(Gentic_df_out.shape[0]):  # Loop through them
                Feat_vec = list(np.array(Features)[np.array(Gentic_df_out.iloc[i, :][Features], dtype=bool)])
                CF_Iter = Find_CF(Search_ID, Lower_Bound, Upper_Bound, Feat_vec)
                CF_Iter = CF_Iter.rename(index={'Counterfactual': 'Counterfactual_{}'.format(i)})
                if i == 0:
                    CF_out = CF_Iter.copy()
                else:
                    CF_out = CF_out.append(
                        CF_Iter.loc['Counterfactual_{}'.format(i), :]).copy()  # Append to output component
            Gentic_df_out.index = CF_out.index[1:]
            # Explain_out['Counterfactual_Table'] = CF_out

            # The rest is just renaming, reformatting and reshaping.
            # To create the format that will be loaded to the DB
            CF_Df_val = pd.DataFrame(
                CF_out.drop(['sk_id_curr', 'target', 'pd', 'predicted_default', 'score'], axis=1).stack()).copy()
            CF_Df_val = CF_Df_val.rename(columns={0: 'Counterfactual_Value'}).copy()
            Case_Df_Val = CF_Df_val.loc['Case', :].reset_index().rename(
                columns={'index': 'feature', 'Counterfactual_Value': 'Case_Value'}).copy()
            CF_Df_val = CF_Df_val.drop('Case').reset_index().rename(
                columns={'level_0': 'CF_ID', 'level_1': 'feature'}).copy()

            CF_Df_bool = pd.DataFrame(
                Gentic_df_out.drop(['Generation', 'Feature_id', 'Score', 'sk_id_curr'], axis=1).stack()).copy()
            CF_Df_bool = CF_Df_bool.reset_index().rename(
                columns={'level_0': 'CF_ID', 'level_1': 'feature', 0: 'CF_Inclusion_Ind'}).copy()
            plot_table = pd.merge(CF_Df_val, Case_Df_Val, on=['feature'], how='inner')
            plot_table = pd.merge(plot_table, CF_Df_bool, on=['feature', 'CF_ID'], how='inner')
            CF_tbl_raw = CF_out[['sk_id_curr', 'score', 'predicted_default']].rename(columns={'score': 'raw_score'})
            CF_tbl_tar = Gentic_df_out[['Score']].rename(columns={'Score': 'target_score'})
            CF_score_tbl = CF_tbl_raw.join(CF_tbl_tar)
            CF_score_tbl['sk_id_curr'] = CF_score_tbl['sk_id_curr'].astype(int)
            Case_Score = CF_score_tbl.loc['Case', 'raw_score']
            CF_score_tbl = CF_score_tbl.reset_index().rename(columns={'index': 'CF_ID'})

            plot_table = pd.merge(plot_table, CF_score_tbl, on=['CF_ID'], how='left')
            plot_table['Case_Score'] = Case_Score
            plot_table.columns = plot_table.columns.str.lower()

            return (plot_table)

        # 05.10 1st generation. Find NUN for all features in isolation
        prog = 0
        printProgressBar(prog, len(Features), prefix='Gen 1 Progress:', suffix='Complete', length=50)
        for feat in Features:
            Gentic_df_i = CF_Score_Calc(Search_ID, Lower_Bound, Upper_Bound, [feat])  # Find NUN and apply
            Gentic_df_i['Feature_id'] = feat
            Gentic_df = Gentic_df.append(Gentic_df_i, ignore_index=True).copy()  # Append to main table
            prog = prog + 1
            printProgressBar(prog, len(Features), prefix='Gen 1 Progress:', suffix='Complete', length=50)

        # If any of the scores meet the specified criteria, outout and then terminate the algorithm
        if (max(Gentic_df['Score']) >= Lower_Bound) & (min(Gentic_df['Score']) <= Upper_Bound):
            return (Out())

        Bool_Vec = np.array(Gentic_df[Features])

        features_np = np.array(Features)
        gen = 2

        # 05.11 Begin a loop that will only terminate when at least one counterfactual is found
        while 1 < 2:
            Cross_Breed_Prev = np.array(Gentic_df[Features])
            Gentic_df['Score_diff'] = abs(
                Gentic_df['Score'] - Target_Score)  # Find the distance between each NUN and the target score
            Gentic_df_i_gen = Gentic_df.loc[Gentic_df['Score_diff'] <= max(
                Gentic_df[['Score_diff']].sort_values('Score_diff', ascending=True).head(N_samples)['Score_diff']),
                              :].copy()  # IMPORTANT!! Take the top n cases that are closest to the target score
            Genetic_bin_vec = np.array(Gentic_df_i_gen[Features])

            # Cross Breed - pair each of the top n counterfactuals with each other remaining feature
            Cross_Breed = np.unique(np.array([i + j for i in Genetic_bin_vec for j in Bool_Vec]), axis=0)
            Cross_Breed = setdiff_nd_positivenums(Cross_Breed, Cross_Breed_Prev)

            Cross_Breed_Word = list(
                [list(features_np[i]) for i in Cross_Breed])  # Crete a word vector for subsetting purposes

            prog = 0
            printProgressBar(prog, len(Cross_Breed_Word), prefix='Gen {} Progress:'.format(gen), suffix='Complete',
                             length=50)

            for feats in Cross_Breed_Word:  # Loop through the feature sets same as before
                Gentic_df_i = CF_Score_Calc(Search_ID, Lower_Bound, Upper_Bound, feats)
                Gentic_df_i['Generation'] = gen
                Gentic_df_i['Feature_id'] = ','.join(feats)
                Gentic_df = Gentic_df.append(Gentic_df_i, ignore_index=True).copy()
                prog = prog + 1
                printProgressBar(prog, len(Cross_Breed_Word), prefix='Gen {} Progress:'.format(gen), suffix='Complete',
                                 length=50)
            gen = gen + 1

            # If any of the scores meet the specified criteria, outout and then terminate the algorithm
            if (max(Gentic_df['Score']) >= Lower_Bound) & (min(Gentic_df['Score']) <= Upper_Bound):
                return (Out())


    # 05.12 Execute the alogorithm
    CF = Explain(args.acc_id, args.target_score, args.lower_bound, args.upper_bound,
                 args.n_samples)  # This is an adjustable parameter, in fact all of them are!!

    # push to DW
    CF.to_sql('plot_table', engine, schema='expl', if_exists='replace')

    # Add primary key, just in case.....
    conn = psycopg2.connect(
        "dbname='hm_crdt' user='{}' password='{}'".format(args.db_user, args.in_password)
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    cur.execute("""
            ALTER TABLE expl.plot_table 
            ADD PRIMARY KEY (CF_ID, feature);
            ;""")
    conn.commit()
    conn.close()
