import psycopg2
import argparse
import pandas as pd
import statsmodels.formula.api as smf
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy import create_engine
from sklearn.neighbors import KNeighborsClassifier

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
        "dbname='hm_crdt' user='{}' password='{}'".format(args.db_user, args.in_password) #args.db_user
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()

    engine = create_engine('postgresql://postgres:{}@localhost:5432/hm_crdt'.format(args.in_password)) #args.db_user,

    conn_test = engine.connect()
    conn_test.close()

    cur.execute("""create schema if not exists misc;""")


    amt_corr_df = pd.read_sql('''select amt_income_total
                    , amt_credit
                    , amt_annuity
                    , amt_goods_price
                    from raw_in.application_train_test
                    ;''', conn)

    amt_corr_df.corr()

    #A function to convert regression to string for imputation
    def reg_2_str(In_Reg):
        reg_str = amt_annuity_reg.params.reset_index().values.tolist()
        reg_str = ['*'.join(map(str,item)) for item in reg_str]
        reg_str = '+'.join(reg_str).replace('Intercept*','')
        return(reg_str)

    #amt_annuity imputation
    amt_annuity_reg_df = pd.read_sql('''select 
                      amt_annuity
                    , amt_credit
                    , amt_income_total
                    from raw_in.application_train_test
                    where amt_annuity is not null
                        and amt_credit is not null
                        and amt_income_total is not null
                    ;''', conn)

    amt_annuity_reg = smf.ols('amt_annuity ~ amt_credit + amt_income_total', data=amt_annuity_reg_df).fit()
    print(amt_annuity_reg.summary())
    amt_annuity_reg_str = reg_2_str(amt_annuity_reg)


    #amt_goods_price imputation
    amt_goods_price_reg_df = pd.read_sql('''select 
                      amt_credit
                    , amt_goods_price
                    from raw_in.application_train_test
                    where amt_credit is not null
                        and amt_credit is not null
                    ;''', conn)

    amt_goods_price_reg = smf.ols('amt_goods_price ~ amt_goods_price', data=amt_goods_price_reg_df).fit()
    print(amt_goods_price_reg.summary())
    amt_goods_price_reg_str = reg_2_str(amt_goods_price_reg)


    #EXT_SOURCE Imputation
    application_train = pd.read_sql('''select * from raw_in.application_train_test where train_test = 'Train';''', conn)
    application_test = pd.read_sql('''select * from raw_in.application_train_test where train_test = 'Test';''', conn)

    #columns_for_modelling = list(set(application_test.dtypes[application_test.dtypes != 'object'].index.tolist())
    #                 - set(['ext_source_1','ext_source_2','ext_source_3','sk_id_curr']))

    #following code lifted from https://medium.com/thecyphy/home-credit-default-risk-part-2-84b58c1ab9d5
    #for ext_col in ['ext_source_2','ext_source_3','ext_source_1']:
        #X_model - datapoints which do not have missing values of given column
        #Y_train - values of column trying to predict with non missing values
        #X_train_missing - datapoints in application_train with missing values
        #X_test_missing - datapoints in application_test with missing values
    #    X_model, X_train_missing, X_test_missing, Y_train = application_train[~application_train[ext_col].isna()][columns_for_modelling], application_train[
    #                                                    application_train[ext_col].isna()][columns_for_modelling], application_test[
    #                                                    application_test[ext_col].isna()][columns_for_modelling], application_train[
    #                                                    ext_col][~application_train[ext_col].isna()]
    #    xg = XGBRegressor(n_estimators = 1000, max_depth = 3, learning_rate = 0.1, n_jobs = -1, random_state = 59)
    #    xg.fit(X_model, Y_train)
    #    application_train[ext_col][application_train[ext_col].isna()] = xg.predict(X_train_missing)
    #    application_test[ext_col][application_test[ext_col].isna()] = xg.predict(X_test_missing)
        #adding the predicted column to columns for modelling for next column's prediction
    #    columns_for_modelling = columns_for_modelling + [ext_col]

    application_train['ext_source_1'] = application_train['ext_source_1'].fillna(application_train['ext_source_1'].mean())
    application_train['ext_source_2'] = application_train['ext_source_2'].fillna(application_train['ext_source_2'].mean())
    application_train['ext_source_3'] = application_train['ext_source_3'].fillna(application_train['ext_source_3'].mean())

    application_test['ext_source_1'] = application_test['ext_source_1'].fillna(application_train['ext_source_1'].mean())
    application_test['ext_source_2'] = application_test['ext_source_2'].fillna(application_train['ext_source_2'].mean())
    application_test['ext_source_3'] = application_test['ext_source_3'].fillna(application_train['ext_source_3'].mean())


    #Transpose
    application_train['CREDIT_ANNUITY_RATIO'] = application_train['amt_credit']/application_train['amt_annuity']
    application_test['CREDIT_ANNUITY_RATIO'] = application_test['amt_credit']/application_test['amt_annuity']


    def neighbors_EXT_SOURCE_feature():
        '''
        Function to generate a feature which contains the means of TARGET of 500 neighbors of a particular row.
        Inputs:
            self
        Returns:
            None
        '''
        #https://www.kaggle.com/c/home-credit-default-risk/discussion/64821
        #imputing the mean of 500 nearest neighbor's target values for each application
        #neighbors are computed using EXT_SOURCE feature and CREDIT_ANNUITY_RATIO
        train_data_for_neighbors = application_train[['ext_source_1','ext_source_2','ext_source_3','CREDIT_ANNUITY_RATIO']].fillna(0)
        train_target = application_train.target
        test_data_for_neighbors = application_test[['ext_source_1','ext_source_2','ext_source_3','CREDIT_ANNUITY_RATIO']].fillna(0)

        knn = KNeighborsClassifier(500, n_jobs = -1)
        knn.fit(train_data_for_neighbors, train_target)
        train_500_neighbors = knn.kneighbors(train_data_for_neighbors)[1]
        test_500_neighbors = knn.kneighbors(test_data_for_neighbors)[1]

        #adding the means of targets of 500 neighbors to new column
        application_train['target_neighbors_500_mean'] = [application_train['target'].iloc[ele].mean() for ele in train_500_neighbors]
        application_test['target_neighbors_500_mean'] = [application_train['target'].iloc[ele].mean() for ele in test_500_neighbors]

    neighbors_EXT_SOURCE_feature()



    EXT_DF = application_train[['ext_source_1','ext_source_2','ext_source_3','target_neighbors_500_mean','sk_id_curr']].append(application_test[['ext_source_1','ext_source_2','ext_source_3','target_neighbors_500_mean','sk_id_curr']])

    EXT_DF.to_sql('ext_source_impute', engine, schema='misc', if_exists='replace')
    cur.execute("""alter table misc.EXT_Source_Impute add primary key (sk_id_curr);""")


    print('Creating ABT')
    cur.execute(
        """
        create schema if not exists ABT;
        
        
        create table ABT.ABT_Full as
        
        select bse.*
            , AMT_CREDIT/bse.AMT_ANNUITY as credit_annuity_ratio
            , AMT_GOODS_PRICE - AMT_CREDIT as credit_down_payment
            , AMT_credit/AMT_GOODS_PRICE as LTV_T0
            , bse.AMT_ANNUITY/AMT_INCOME_TOTAL as PAYMENT_INCOME_RATIO
            , bse.AMT_CREDIT/AMT_INCOME_TOTAL as Credit_Income_Ratio
            
            
            , ext.ext_source_1
            , ext.ext_source_2
            , ext.ext_source_3
            , ext_source_min
            , ext_source_max
            
            , ext_source_max - ext_source_min as ext_source_spread
            , target_neighbors_500_mean
            
            , ext.ext_source_1^2 as ext_source_1_Squared
            , ext.ext_source_2^2 as ext_source_2_Squared
            , ext.ext_source_3^2 as ext_source_3_Squared
            , ext.ext_source_1*ext.ext_source_2 as ext_source_1x2
            , ext.ext_source_1*ext.ext_source_3 as ext_source_1x3
            , ext.ext_source_2*ext.ext_source_3 as ext_source_2x3
            , ext.ext_source_1*ext.ext_source_2*ext.ext_source_3 as ext_source_1x2x3
            , (AMT_CREDIT/bse.AMT_ANNUITY)*ext_source_1 as credit_annuity_ratio_x_ext_source_1
            , (AMT_CREDIT/bse.AMT_ANNUITY)*ext_source_2 as credit_annuity_ratio_x_ext_source_2
            , (AMT_CREDIT/bse.AMT_ANNUITY)*ext_source_3 as credit_annuity_ratio_x_ext_source_3
            , (bse.AMT_ANNUITY/AMT_INCOME_TOTAL)*ext_source_1 as PAYMENT_INCOME_RATIO_x_ext_source_1
            , (bse.AMT_ANNUITY/AMT_INCOME_TOTAL)*ext_source_2 as PAYMENT_INCOME_RATIO_x_ext_source_2
            , (bse.AMT_ANNUITY/AMT_INCOME_TOTAL)*ext_source_3 as PAYMENT_INCOME_RATIO_x_ext_source_3
            , (bse.AMT_CREDIT/AMT_INCOME_TOTAL)*ext_source_1 as Credit_Income_Ratio_x_ext_source_1
            , (bse.AMT_CREDIT/AMT_INCOME_TOTAL)*ext_source_2 as Credit_Income_Ratio_x_ext_source_2
            , (bse.AMT_CREDIT/AMT_INCOME_TOTAL)*ext_source_3 as Credit_Income_Ratio_x_ext_source_3
            , (AMT_credit/AMT_GOODS_PRICE)*ext_source_1 as LTV_T0_x_ext_source_1
            , (AMT_credit/AMT_GOODS_PRICE)*ext_source_2 as LTV_T0_x_ext_source_2
            , (AMT_credit/AMT_GOODS_PRICE)*ext_source_3 as LTV_T0_x_ext_source_3
            , ext.ext_source_1*days_birth as ext_source_1xAge
            , ext.ext_source_2*days_birth as ext_source_2xAge
            , ext.ext_source_3*days_birth as ext_source_3xAge
        
            
            , coalesce(Num_Unknown,0) as Num_Unknown
            , coalesce(Num_Healthy,0) as Num_Healthy
            , coalesce(Num_Delinquent,0) as Num_Delinquent
            , coalesce(num_write_off,0) as num_write_off
            , coalesce(num_dist_credit_type,0) as num_dist_credit_type
            , coalesce(Num_Unknown/Num_months,0) as prop_unknown
            , coalesce(Num_Healthy/Num_months,0) as prop_healthy
            , coalesce(num_delinquent/Num_months,0) as prop_delinquent
            , coalesce(num_write_off/Num_months,0) as prop_write_off
            , coalesce(num_dist_credit_type/Num_months,0) as prop_dist_credit_type
            , coalesce(num_months,0) as num_months
            --, most_recent_delinquent
            --, most_recent_write_off
            
            , coalesce(Num_On_Time,0) as Num_On_Time
            , coalesce(Num_Late,0) as Num_Late
            , coalesce(Num_Partial_Payments,0) as Num_Partial_Payments
            , coalesce(Num_Overpayments,0) as Num_Overpayments
            , coalesce(Num_On_Time/num_installments,0) as Prop_on_time
            , coalesce(Num_Late/num_installments,0) as Prop_Late
            , coalesce(Num_Partial_Payments/num_installments,0) as Prop_Partial_Payments
            , coalesce(Num_Overpayments/num_installments,0) as Prop_Overpayments
            , coalesce(variance_payment,0) as variance_payment
            , coalesce(var_payment_spread,0) as var_payment_spread
            
            
            --Credit Card
            , coalesce(num_cc,0) as num_cc
            , coalesce(num_cc_installments/num_cc,0) as prop_cc_installments
            , coalesce(num_max_credit_months,0) as num_max_credit_months
            , coalesce(avg_bal_limit_ratio,0) as avg_bal_limit_ratio
            , coalesce(bal_limit_ratio,0) as bal_limit_ratio
            , coalesce(avg_dpd,0) as avg_dpd
            , coalesce(Total_dpd,0) as Total_dpd
            , coalesce(num_missed_cc_ins/num_cc_payments,0) as prop_missed_cc_install
            , coalesce(TOTAL_PAYMENT,0) as TOTAL_INSTALMENT
            , coalesce(CC_Balance,0) as CC_Balance
            , coalesce(cc_amt_drawings_atm_current,0) as cc_amt_drawings_atm_current
            , coalesce(cc_amt_drawings_current,0) as cc_amt_drawings_current
            , coalesce(cc_amt_payment_current,0) as cc_amt_payment_current
            , coalesce(cc_amt_payment_current,0)/(case when coalesce(CC_Balance,0) = 0 then 1 else coalesce(CC_Balance,0) end) as CC_Payment_Bal_Ratio
            , coalesce(CC_Balance,0)/bse.AMT_CREDIT as CC_Bal_Credit_Ratio
            , coalesce(CC_Balance,0)/bse.AMT_INCOME_TOTAL as CC_Bal_Income_Ratio
    
    
            -- bureau
            , coalesce(num_bur_accs,0) as num_bur_accs
            , coalesce(bur.num_closed,0) as num_closed_bur
            , coalesce(bur.num_past_end_Date,0) as num_past_end_Date_bur
            , coalesce(num_closed/num_bur_accs,0) as prop_past_loans_closed
            , coalesce(num_past_end_Date/num_bur_accs,0) as prop_past_end_Date
            , coalesce(AMT_CREDIT_SUM_OVERDUE,0) as AMT_CREDIT_SUM_OVERDUE_bur
            , coalesce(AMT_CREDIT_SUM_DEBT,0) as AMT_CREDIT_SUM_DEBT_bur
            , coalesce(AMT_CREDIT_SUM,0) as AMT_CREDIT_SUM_bur
            , coalesce(bur.AMT_ANNUITY,0) as AMT_ANNUITY_bur
            , coalesce(num_dist_credit_bur,0) as num_dist_credit_bur
            , coalesce(AVG_CNT_CREDIT_PROLONG,0) as AVG_CNT_CREDIT_PROLONG_bur
            , coalesce(AMT_CREDIT_SUM_DEBT/(case when AMT_CREDIT_SUM_OVERDUE = 0 then 1 else AMT_CREDIT_SUM_OVERDUE end),0) as OVERDUE_DEBT_RATIO_bur
            , coalesce(AMT_CREDIT_SUM_DEBT/(case when AMT_CREDIT_SUM = 0 then 1 else AMT_CREDIT_SUM end),0) as DEBT_CREDIT_RATIO_bur
            , coalesce(max_DAYS_CREDIT,0) as max_DAYS_CREDIT_bur
            , coalesce(min_DAYS_CREDIT,0) as min_DAYS_CREDIT_bur
            , coalesce(avg_DAYS_CREDIT,0) as avg_DAYS_CREDIT_bur
            , coalesce(max_days_credit_enddate,0) as max_days_credit_enddate_bur
            , coalesce(min_days_credit_enddate,0) as min_days_credit_enddate_bur
            , coalesce(max_days_enddate_fact,0) as max_days_enddate_fact
            , coalesce(min_days_enddate_fact,0) as min_days_enddate_fact
    
    
            --bureau combined with other stuff
            , bse.AMT_ANNUITY/bse.AMT_ANNUITY as AMT_ANNUITY_bur_ratio
            , bse.AMT_ANNUITY/bse.AMT_CREDIT as AMT_CREDIT_bur_ratio
            , coalesce(AMT_CREDIT_SUM_OVERDUE,0)/bse.AMT_ANNUITY as AMT_ANNUITY_CS_OVERDUE_ratio_bur
            , coalesce(AMT_CREDIT_SUM_DEBT,0)/bse.AMT_ANNUITY as AMT_ANNUITY_CS_DEBT_ratio_bur
            , coalesce(AMT_CREDIT_SUM_OVERDUE,0)/bse.AMT_ANNUITY as AMT_ANNUITY_CS_OVERDUE_ratio_bse
            , coalesce(AMT_CREDIT_SUM_DEBT,0)/bse.AMT_ANNUITY as AMT_ANNUITY_CS_DEBT_ratio_bse
            , bse.AMT_ANNUITY/bse.AMT_GOODS_PRICE as Goods_price_annu_bur_ratio
            , coalesce(AMT_CREDIT_SUM_OVERDUE,0)/bse.AMT_GOODS_PRICE as Goods_price_amt_csod_bur_ratio
            , coalesce(AMT_CREDIT_SUM_DEBT,0)/bse.AMT_GOODS_PRICE as Goods_price_amt_cs_bur_ratio
    
    
            --POS Cash balance
            , coalesce(num_POS_payments,0) as num_POS_payments
            , coalesce(POS_DPD_Month,0) as POS_DPD_Month
            , coalesce(POS_DPD_def_Month,0) as POS_DPD_def_Month
            , coalesce(cast(POS_DPD_Month as float)/cast(num_POS_payments as float),0) as POS_DPD_Month_ratio
            , coalesce(cast(POS_DPD_def_Month as float)/cast(num_POS_payments as float),0) as POS_DPD_def_Month_ratio
            , coalesce(POS_Completed,0) as POS_Completed
            , coalesce(cast(POS_Completed as float)/cast(num_POS_payments as float),0) as POS_Completed_ratio
            , coalesce(POS_time_since_last_dpd,0) as POS_time_since_last_dpd
            
    
            --Previous application
            , case when AMT_GOODS_PRICE_PREV = 0 then 1
                else coalesce(AMT_CREDIT_PREV/AMT_GOODS_PRICE_PREV,0)
                end as LTV_Prev
            , case when AMT_CREDIT_PREV = 0 then 0
                else coalesce(AMT_APPLICATION_PREV/AMT_CREDIT_PREV,0)
                end as Application_Credit_Ratio_Prev
            , case when AMT_CREDIT_PREV = 0 then 0
                else coalesce(AMT_DOWN_PAYMENT_PREV/AMT_CREDIT_PREV,0) 
                end as Credit_Downpayment_Ratio_Prev
            , coalesce(num_prev,0) as num_prev
            , coalesce(num_credit_gt_app,0) as num_credit_gt_app
            , coalesce(num_credit_gt_app/num_prev,0) as prop_credit_gt_app
            , coalesce(num_unsecured,0) as num_unsecured
            , coalesce(num_unsecured/num_prev,0) as prop_unsecured
            , case when coalesce(AMT_CREDIT_PREV,0) = 0 then 0
                else coalesce(AMT_CREDIT_Unsecured/AMT_CREDIT_PREV,0)
                end as Prop_Credit_Unsecured
                
            , coalesce(num_Approved,0) as num_Approved_prev
            , coalesce(num_Canceled,0) as num_Canceled_prev
            , coalesce(num_Refused,0) as num_Refused_prev
            , coalesce(num_Unused,0) as num_Unused_prev
            , coalesce(num_Approved/num_prev,0) as prop_Approved_prev
            , coalesce(num_Canceled/num_prev,0) as prop_Canceled_prev
            , coalesce(num_Refused/num_prev,0) as prop_Refused_prev
            , coalesce(num_Unused/num_prev,0) as prop_Unused_prev
            , coalesce(Num_payment_type_Bank_Cash,0) as Num_payment_type_Bank_Cash_prev
            , coalesce(Num_payment_type_employer_Cashless,0) as Num_payment_type_employer_Cashless_prev
            , coalesce(Num_payment_type_Non_Cash,0) as Num_payment_type_Non_Cash_prev
            , coalesce(Num_payment_type_XNA,0) as Num_payment_type_XNA_prev
            , coalesce(Num_payment_type_Bank_Cash/num_prev,0) as prop_payment_type_Bank_Cash_prev
            , coalesce(Num_payment_type_employer_Cashless/num_prev,0) as prop_payment_type_employer_Cashless_prev
            , coalesce(Num_payment_type_Non_Cash/num_prev,0) as prop_payment_type_Non_Cash_prev
            , coalesce(Num_payment_type_XNA/num_prev,0) as prop_payment_type_XNA_prev
            , coalesce(DAYS_DECISION_avg_prev,0) as DAYS_DECISION_avg_prev
            , coalesce(DAYS_DECISION_max_prev,0) as DAYS_DECISION_max_prev
            , coalesce(num_type_suite_null,0) as num_type_suite_null_prev
            , coalesce(num_type_suite_Children,0) as num_type_suite_Children_prev
            , coalesce(num_type_suite_Family,0) as num_type_suite_Family_prev
            , coalesce(num_type_suite_Other,0) as num_type_suite_Other_prev
            , coalesce(num_type_suite_spouse,0) as num_type_suite_spouse_prev
            , coalesce(num_type_suite_Unaccompanied,0) as num_type_suite_Unaccompanied_prev
            , coalesce(num_type_suite_null/num_prev,0) as prop_type_suite_null_prev
            , coalesce(num_type_suite_Children/num_prev,0) as prop_type_suite_Children_prev
            , coalesce(num_type_suite_Family/num_prev,0) as prop_type_suite_Family_prev
            , coalesce(num_type_suite_Other/num_prev,0) as prop_type_suite_Other_prev
            , coalesce(num_type_suite_spouse/num_prev,0) as prop_type_suite_spouse_prev
            , coalesce(num_type_suite_Unaccompanied/num_prev,0) as prop_type_suite_Unaccompanied_prev
            , coalesce(num_client_type_New,0) as num_client_type_New_prev
            , coalesce(num_client_type_Refreshed,0) as num_client_type_Refreshed_prev
            , coalesce(num_client_type_Repeater,0) as num_client_type_Repeater_prev
            , coalesce(num_client_type_XNA,0) as num_client_type_XNA_prev
            , coalesce(num_client_type_New/num_prev,0) as prop_client_type_New_prev
            , coalesce(num_client_type_Refreshed/num_prev,0) as prop_client_type_Refreshed_prev
            , coalesce(num_client_type_Repeater/num_prev,0) as prop_client_type_Repeater_prev
            , coalesce(num_client_type_XNA/num_prev,0) as prop_client_type_XNA_prev
            
            , coalesce(num_portfolio_Cards,0) as num_portfolio_Cards_prev
            , coalesce(num_portfolio_Cars,0) as num_portfolio_Cars_prev
            , coalesce(num_portfolio_Cash,0) as num_portfolio_Cash_prev
            , coalesce(num_portfolio_POS,0) as num_portfolio_POS_prev
            , coalesce(num_portfolio_XNA,0) as num_portfolio_XNA_prev
            , coalesce(num_portfolio_Cards/num_prev,0) as prop_portfolio_Cards_prev
            , coalesce(num_portfolio_Cars/num_prev,0) as prop_portfolio_Cars_prev
            , coalesce(num_portfolio_Cash/num_prev,0) as prop_portfolio_Cash_prev
            , coalesce(num_portfolio_POS/num_prev,0) as prop_portfolio_POS_prev
            , coalesce(num_portfolio_XNA/num_prev,0) as prop_portfolio_XNA_prev
        
            , coalesce(num_product_type_XNA,0) as num_product_type_XNA_prev
            , coalesce(num_product_type_Walk_in,0) as num_product_type_Walk_in_prev
            , coalesce(num_product_type_X_Sell,0) as num_product_type_X_Sell_prev
            , coalesce(num_product_type_XNA/num_prev,0) as prop_product_type_XNA_prev
            , coalesce(num_product_type_Walk_in/num_prev,0) as prop_product_type_Walk_in_prev
            , coalesce(num_product_type_X_Sell/num_prev,0) as prop_product_type_X_Sell_prev
        
            , coalesce(num_channel_type_AP ,0) as num_channel_type_AP_prev
            , coalesce(num_channel_type_Car_dealer ,0) as num_channel_type_Car_dealer_prev
            , coalesce(num_channel_type_Corp_sales ,0) as num_channel_type_Corp_sales_prev
            , coalesce(num_channel_type_Contact_center ,0) as num_channel_type_Contact_center_prev
            , coalesce(num_channel_type_Credit_and_cash_offices ,0) as num_channel_type_Credit_and_cash_offices_prev
            , coalesce(num_channel_type_Regional_Local ,0) as num_channel_type_Regional_Local_prev
            , coalesce(num_channel_type_country_wide ,0) as num_channel_type_country_wide_prev
            , coalesce(num_channel_type_Stone ,0) as num_channel_type_Stone_prev
            , coalesce(num_channel_type_AP/num_prev ,0) as prop_channel_type_AP_prev
            , coalesce(num_channel_type_Car_dealer/num_prev ,0) as prop_channel_type_Car_dealer_prev
            , coalesce(num_channel_type_Corp_sales/num_prev ,0) as prop_channel_type_Corp_sales_prev
            , coalesce(num_channel_type_Contact_center/num_prev ,0) as prop_channel_type_Contact_center_prev
            , coalesce(num_channel_type_Credit_and_cash_offices/num_prev ,0) as prop_channel_type_Credit_and_cash_offices_prev
            , coalesce(num_channel_type_Regional_Local/num_prev ,0) as prop_channel_type_Regional_Local_prev
            , coalesce(num_channel_type_country_wide/num_prev ,0) as prop_channel_type_country_wide_prev
            , coalesce(num_channel_type_Stone/num_prev ,0) as prop_channel_type_Stone_prev
            
            , coalesce(max_sellerplace_area,0) as max_sellerplace_area_prev
            
            , coalesce(num_yield_group_high,0) as num_yield_group_high_prev
            , coalesce(num_yield_group_low_action,0) as num_yield_group_low_action_prev
            , coalesce(num_yield_group_low_normal,0) as num_yield_group_low_normal_prev
            , coalesce(num_yield_group_middle,0) as num_yield_group_middle_prev
            , coalesce(num_yield_group_XNA,0) as num_yield_group_XNA_prev
            , coalesce(num_yield_group_high/num_prev,0) as prop_yield_group_high_prev
            , coalesce(num_yield_group_low_action/num_prev,0) as prop_yield_group_low_action_prev
            , coalesce(num_yield_group_low_normal/num_prev,0) as prop_yield_group_low_normal_prev
            , coalesce(num_yield_group_middle/num_prev,0) as prop_yield_group_middle_prev
            , coalesce(num_yield_group_XNA/num_prev,0) as prop_yield_group_XNA_prev
    
            , coalesce(num_insured,0) as num_insured_prev
            , coalesce(num_insured/num_prev,0) as prev_insured_prev
    
    
            , coalesce(avg_rate_interest_primary_prev,0) AS avg_rate_interest_primary_prev
            , coalesce(avg_rate_interest_privelaged_prev,0) as avg_rate_interest_privelaged_prev
            , coalesce(max_rate_interest_primary_prev,0) as max_rate_interest_primary_prev
            , coalesce(max_rate_interest_privelaged_prev,0) as max_rate_interest_privelaged_prev
            , coalesce(min_rate_interest_primary_prev,0) as min_rate_interest_primary_prev
            , coalesce(min_rate_interest_privelaged_prev,0) as min_rate_interest_privelaged_prev
    
            , coalesce(max_rate_interest_primary_prev - min_rate_interest_primary_prev,0) as spread_rate_interest_primary_prev
            , coalesce(max_rate_interest_privelaged_prev - min_rate_interest_privelaged_prev,0) as spread_rate_interest_privelaged_prev
    
            , coalesce(days_last_due_1st_version_avg_prev,0) as days_last_due_1st_version_avg_prev
            , coalesce(days_last_due_1st_version_max_prev,0) as days_last_due_1st_version_max_prev
            , coalesce(days_last_due_1st_version_min_prev,0) as days_last_due_1st_version_min_prev
    
            , coalesce(days_first_due_avg_prev,0) as days_first_due_avg_prev
            , coalesce(days_first_due_max_prev,0) as days_first_due_max_prev
            , coalesce(days_first_due_min_prev,0) as days_first_due_min_prev
    
            , coalesce(days_last_due_avg_prev,0) as days_last_due_avg_prev
            , coalesce(days_last_due_max_prev,0) as days_last_due_max_prev
            , coalesce(days_last_due_min_prev,0) as days_last_due_min_prev
            
            , coalesce(days_last_due_max_prev,0) - coalesce(days_first_due_min_prev,0) as due_spread
    
            , coalesce(days_first_drawing_avg_prev,0) as days_first_drawing_avg_prev
            , coalesce(days_first_drawing_max_prev,0) as days_first_drawing_max_prev
            , coalesce(days_first_drawing_min_prev,0) as days_first_drawing_min_prev
                      
            , coalesce(days_termination_avg_prev,0) as days_termination_avg_prev
            , coalesce(days_termination_max_prev,0) as days_termination_max_prev
            , coalesce(days_termination_min_prev,0) as days_termination_min_prev
    
            --Previous application to current ratios
            , case when coalesce(num_prev,0) = 0 or coalesce(AMT_GOODS_PRICE_PREV,0) = 0 then 1
                else AMT_CREDIT/(AMT_CREDIT_PREV/num_prev) 
                end as AMT_CREDIT_PREV_Ratio
            , case when coalesce(num_prev,0) = 0 or coalesce(AMT_GOODS_PRICE_PREV,0) = 0 then 1
                else AMT_GOODS_PRICE/(AMT_GOODS_PRICE_PREV/num_prev) 
                end as AMT_GOODS_PRICE_PREV_Ratio
            , case when coalesce(num_prev,0) = 0 or coalesce(AMT_DOWN_PAYMENT_PREV,0) = 0 then 1
                else (AMT_GOODS_PRICE - AMT_CREDIT)/(AMT_DOWN_PAYMENT_PREV/num_prev) 
                end as Down_Payment_Prev_Ratio
                
            --
            , case when coalesce(AMT_CREDIT_PREV,0) = 0 then 0
                else coalesce(coalesce(TOTAL_PAYMENT,0)/AMT_CREDIT_PREV,0)
                end as Payment_Credit_Ratio
            
        from (
        select SK_ID_CURR
            , TARGET
            , AMT_INCOME_TOTAL as AMT_INCOME_TOTAL
            , AMT_CREDIT as AMT_CREDIT
            , coalesce(AMT_ANNUITY,{amt_annuity_reg}) as AMT_ANNUITY
            , coalesce(AMT_GOODS_PRICE,{amt_goods_price_reg}) as AMT_GOODS_PRICE
            , days_birth
            , flag_own_Car
            , code_gender
            , flag_own_realty
            , cnt_children
            , coalesce(name_type_suite,'Unidentified') as name_type_suite
            , case when name_income_type in ('Businessman','Maternity leave','Student','Unemployed') then 'No stable income'
                else name_income_type
                end as name_income_type
            , name_education_Type
            , case when name_family_status = 'Unknown' then 'Single / not married'
                else name_family_status
                end as name_family_status
            , region_population_relative
            , case when days_employed > 0 then 0 
                else days_employed
                end as days_employed
                
            , cast(days_employed as float)/cast(days_birth as float) as Employment_ratio
            , case when days_registration > 0 then 0 
                else days_registration
                end as days_registration
            , case when days_id_publish > 0 then 0 
                else days_id_publish
                end as days_id_publish
            , coalesce(own_Car_age,0) as own_Car_age
            , flag_work_phone
            , flag_phone
            , flag_email
            , coalesce(occupation_type,'Unidentified') as occupation_type
            , case when cnt_fam_members >= 5 then '5+'
                else cast(coalesce(cnt_fam_members,1) as varchar(2))
                end as cnt_fam_members
            , region_rating_client
            , case when region_rating_client_w_city < 0 then 1
                else region_rating_client_w_city
                end as region_rating_client_w_city
            , weekday_appr_process_start
            , hour_appr_process_start
            , reg_region_not_live_region
            , reg_city_not_live_city
            , reg_city_not_work_city
            , case 
                    when ORganization_type 
                        in ('Trade: type 4','Industry: type 12','Transport: type 1','Trade: type 6') 
                        then 'Organization Type 1'
                    when ORganization_type 
                        in ('Security Ministries','University','Police','Military','Bank','XNA','Culture') 
                        then 'Organization Type 2'
                    when ORganization_type 
                        in ('Insurance','Religion','School','Trade: type 5','Industry: type 10','Services','Hotel','Electricity','Medicine','Industry: type 6','Industry: type 9','Industry: type 5') 
                        then 'Organization Type 3'
                    when ORganization_type 
                        in ('Government','Kindergarten','Emergency','Industry: type 2','Trade: type 2','Transport: type 2','Telecom','Other','Legal Services','Industry: type 7','Housing')
                        then 'Organization Type 4'
                    when ORganization_type 
                        in ('Advertising','Business Entity Type 1','Postal','Business Entity Type 2','Industry: type 11','Trade: type 1','Transport: type 4')
                        then 'Organization Type 5'
                    when ORganization_type 
                        in ('Mobile','Business Entity Type 3','Trade: type 7','Industry: type 4','Security','Self-employed','Trade: type 3','Realtor')
                        then 'Organization Type 6'
                    when ORganization_type 
                        in ('Industry: type 3','Agriculture','Industry: type 1','Cleaning','Construction','Restaurant')
                        then 'Organization Type 7'
                    else 'Organization Type 8'
                    end as ORganization_type
            , coalesce(obs_30_cnt_social_circle,0) as obs_30_cnt_social_circle
            , coalesce(obs_60_cnt_social_circle,0) as obs_60_cnt_social_circle
            , coalesce(def_30_cnt_social_circle,0) as def_30_cnt_social_circle
            , coalesce(def_60_cnt_social_circle,0) as def_60_cnt_social_circle
            , coalesce(def_30_cnt_social_circle/(case when obs_30_cnt_social_circle = 0 then 1 else obs_30_cnt_social_circle end), 0) as def_30_ratio_social_circle
            , coalesce(def_60_cnt_social_circle/(case when obs_60_cnt_social_circle = 0 then 1 else obs_60_cnt_social_circle end), 0) as def_60_ratio_social_circle
            , coalesce(days_last_phone_change/-365.25,0) as years_last_phone_change
            
            , case when AMT_INCOME_TOTAL > AMT_CREDIT then 1
                else 0
                end as Income_GT_Credit_Flag
                
        from raw_in.application_train_test
        ) as bse
        
        left join (select bse.*
                       , case when ext_source_1 > ext_source_2 and ext_source_1 > ext_source_3 then ext_source_1
                               when ext_source_2 > ext_source_3 and ext_source_2 > ext_source_3 then ext_source_2
                               when ext_source_3 > ext_source_1 and ext_source_3 > ext_source_2 then ext_source_3
                               end as ext_source_max
                               
                       , case when ext_source_1 < ext_source_2 and ext_source_1 < ext_source_3 then ext_source_1
                               when ext_source_2 < ext_source_3 and ext_source_2 < ext_source_3 then ext_source_2
                               when ext_source_3 < ext_source_1 and ext_source_3 < ext_source_2 then ext_source_3
                               end as ext_source_min  
                   
                   from misc.ext_source_impute as bse
                   
                   ) as ext
            on bse.sk_id_curr = ext.sk_id_curr
            
        left join (select bse.sk_id_curr
                       , cast(count(*) as float) as num_months
                       , cast(count(distinct credit_type) as float) as num_dist_credit_type
                       , cast(sum(case when STATUS = 'X' then 1 else 0 end) as float) as Num_Unknown
                       , cast(sum(case when STATUS = '0' then 1 else 0 end) as float) as Num_Healthy
                       , cast(sum(case when STATUS not in ('0','X','C','5') then 1 else 0 end) as float) as num_delinquent
                       , cast(sum(case when STATUS = '5' then 1 else 0 end) as float) as num_write_off
                       --, max(case when STATUS not in ('0','X','C','5') then MONTHS_BALANCE end) as float) as most_recent_delinquent
                       --, max(case when STATUS = '5' then MONTHS_BALANCE end) as float) as most_recent_write_off
                       
                   from raw_in.bureau as bse
                   left join raw_in.bureau_balance as bal
                       on bse.SK_ID_BUREAU = bal.SK_ID_BUREAU
                       group by 1
                   ) as bur_bal
                    on bse.sk_id_curr = bur_bal.sk_id_curr
    
    
          --features got from https://www.kaggle.com/shanth84/home-credit-bureau-data-feature-engineering       
        left join (select sk_id_curr
                       , cast(count(distinct bse.SK_ID_BUREAU) as float) as num_bur_accs
                       , count(distinct credit_type) as num_dist_credit_bur
                       , cast(sum(case when CREDIT_ACTIVE = 'Closed' then 1 else 0 end) as float) as num_closed
                       , cast(sum(case when days_credit_enddate < 0 then 0 else 1 end) as float) as num_past_end_Date
                       , cast(avg(coalesce(CNT_CREDIT_PROLONG,0)) as float) as AVG_CNT_CREDIT_PROLONG
                       , cast(sum(coalesce(AMT_CREDIT_SUM_OVERDUE,0)) as float) as AMT_CREDIT_SUM_OVERDUE
                       , cast(sum(coalesce(AMT_CREDIT_SUM_DEBT,0)) as float) as AMT_CREDIT_SUM_DEBT
                       , cast(sum(coalesce(AMT_CREDIT_SUM,0)) as float) as AMT_CREDIT_SUM
                       , cast(sum(coalesce(AMT_ANNUITY,0)) as float) as AMT_ANNUITY
                       , max(DAYS_CREDIT) as max_DAYS_CREDIT
                       , min(DAYS_CREDIT) as min_DAYS_CREDIT
                       , avg(DAYS_CREDIT) as avg_DAYS_CREDIT
                       , max(days_credit_enddate) as max_days_credit_enddate
                       , min(days_credit_enddate) as min_days_credit_enddate
                       , max(days_enddate_fact) as max_days_enddate_fact
                       , min(days_enddate_fact) as min_days_enddate_fact
                       
                    from raw_in.bureau as bse
                    group by sk_id_curr
                   ) as bur
                    on bse.sk_id_curr = bur.sk_id_curr
    
        --some features lifted from https://www.kaggle.com/shanth84/credit-card-balance-feature-engineering/
        left join (select bse.sk_id_curr
                       , cast(count(distinct bse.SK_ID_PREV) as float) as num_cc
                       , cast(sum(case when amt_balance > amt_credit_limit_actual then 1 else 0 end) as float) as num_max_credit_months
                       , avg(amt_balance/cast((case when amt_credit_limit_actual = 0 then 1 else amt_credit_limit_actual end) as float)) as bal_limit_ratio
                       , cast(sum(num_cc_installments) as float) as num_cc_installments
                       , cast(count(*) as float) as num_cc_payments
                       , avg(max_bal_limit_ratio) as avg_bal_limit_ratio
                       , avg(num_dpd) as avg_dpd
                       , cast(sum(case when sk_DPD != 0 then 1 else 0 end) as float) as Total_dpd
                       , cast(sum(case when AMT_INST_MIN_REGULARITY > AMT_PAYMENT_CURRENT then 1 else 0 end) as float) as num_missed_cc_ins
                       , sum(coalesce(amt_balance,0)) as CC_Balance
                       , sum(coalesce(amt_drawings_atm_current,0)) as cc_amt_drawings_atm_current
                       , sum(coalesce(amt_drawings_current,0)) as cc_amt_drawings_current
                       , sum(coalesce(amt_payment_current,0)) as cc_amt_payment_current
                    
                      from raw_in.credit_card_balance as bse
                      left join (select sk_id_curr
                                 , sk_id_prev
                                 , cast(max(CNT_INSTALMENT_MATURE_CUM) as float) as num_cc_installments
                                 , max(AMT_BALANCE)/cast(max(case when AMT_CREDIT_LIMIT_ACTUAL = 0 then 1 else AMT_CREDIT_LIMIT_ACTUAL end) as float) as max_bal_limit_ratio
                                 , cast(sum(case when sk_DPD != 0 then 1 else 0 end) as float) as num_dpd
                                 
                                 from raw_in.credit_card_balance
                                 group by sk_id_curr
                                 , sk_id_prev
                          ) as prev
                      on bse.sk_id_curr = prev.sk_id_curr
                      and bse.sk_id_prev = prev.sk_id_prev
                      
                      group by bse.sk_id_curr
                      ) as cc
                    on bse.sk_id_curr = cc.sk_id_curr
    
    
        left join (select sk_id_curr
                       , cast(count(*) as float) as num_installments
                       , cast(sum(case when DAYS_INSTALMENT = DAYS_ENTRY_PAYMENT then 1 else 0 end) as float)  as Num_On_Time
                       , cast(sum(case when DAYS_INSTALMENT < DAYS_ENTRY_PAYMENT then 1 else 0 end) as float)  as Num_Late
                       , cast(sum(case when AMT_INSTALMENT > AMT_PAYMENT then 1 else 0 end) as float) as Num_Partial_Payments
                       , cast(sum(case when AMT_INSTALMENT < AMT_PAYMENT then 1 else 0 end) as float) as Num_Overpayments
                       , sum(AMT_PAYMENT) as TOTAL_PAYMENT
                       , stddev(AMT_PAYMENT) as variance_payment
                       , stddev(AMT_PAYMENT - AMT_INSTALMENT) as var_payment_spread
                   
                    from raw_in.installments_payments
                    group by 1) as ins
                    on bse.sk_id_curr = ins.sk_id_curr
                    
        left join (select sk_id_curr
                       , count(*) as num_POS_payments
                       , sum(case when sk_dpd > 0 then 1 else 0 end) as POS_DPD_Month
                       , sum(case when sk_dpd_def > 0 then 1 else 0 end) as POS_DPD_def_Month
                       , sum(case when name_contract_status = 'Completed' then 1 else 0 end) as POS_Completed
                       , min(case when sk_dpd > 0 then cnt_instalment end) as POS_time_since_last_dpd
                       
                   from raw_in.pos_cash_balance
                   group by 1
                ) as POS
                on bse.sk_id_curr = POS.sk_id_curr
                
        left join (select sk_id_curr
                      , cast(count(*) as float) as num_prev
                      , sum(AMT_CREDIT) as AMT_CREDIT_PREV
                      , sum(AMT_GOODS_PRICE) as AMT_GOODS_PRICE_PREV
                      , sum(AMT_APPLICATION) as AMT_APPLICATION_PREV
                      , cast(sum(case when AMT_CREDIT >= AMT_APPLICATION then 1 else 0 end) as float) as num_credit_gt_app
                      , sum(AMT_DOWN_PAYMENT) as AMT_DOWN_PAYMENT_PREV
                      , sum(case when AMT_GOODS_PRICE is null or AMT_GOODS_PRICE = 0 then 1 else 0 end) as num_unsecured
                      , sum(case when AMT_GOODS_PRICE is null or AMT_GOODS_PRICE = 0 then AMT_CREDIT else 0 end) as AMT_CREDIT_Unsecured
                      
                      , cast(avg(rate_interest_primary) as float) as avg_rate_interest_primary_prev
                      , cast(avg(rate_interest_privileged) as float) as avg_rate_interest_privelaged_prev
                      , cast(max(rate_interest_primary) as float) as max_rate_interest_primary_prev
                      , cast(max(rate_interest_privileged) as float) as max_rate_interest_privelaged_prev
                      , cast(min(rate_interest_primary) as float) as min_rate_interest_primary_prev
                      , cast(min(rate_interest_privileged) as float) as min_rate_interest_privelaged_prev
                      
                      , cast(sum(case when name_contract_status = 'Approved' then 1 else 0 end) as float) as num_Approved
                      , cast(sum(case when name_contract_status = 'Canceled' then 1 else 0 end) as float) as num_Canceled
                      , cast(sum(case when name_contract_status = 'Refused' then 1 else 0 end) as float) as num_Refused
                      , cast(sum(case when name_contract_status = 'Unused offer' then 1 else 0 end) as float) as num_Unused
                      
                      , cast(sum(case when name_payment_type = 'Cash through the bank' then 1 else 0 end) as float) as Num_payment_type_Bank_Cash
                      , cast(sum(case when name_payment_type = 'Cashless from the account of the employer' then 1 else 0 end) as float) as Num_payment_type_employer_Cashless
                      , cast(sum(case when name_payment_type = 'Non-cash from your account' then 1 else 0 end) as float) as Num_payment_type_Non_Cash
                      , cast(sum(case when name_payment_type = 'XNA' then 1 else 0 end) as float) as Num_payment_type_XNA
                      
                      , avg(cast(DAYS_DECISION as float)*-1) as DAYS_DECISION_avg_prev
                      , cast(min(DAYS_DECISION*-1) as float) as DAYS_DECISION_min_prev
                      , cast(max(DAYS_DECISION*-1) as float) as DAYS_DECISION_max_prev
                      
                      , cast(sum(case when name_type_suite is null then 1 else 0 end) as float) as num_type_suite_null
                      , cast(sum(case when name_type_suite = 'Children' then 1 else 0 end) as float) as num_type_suite_Children
                      , cast(sum(case when name_type_suite = 'Family' then 1 else 0 end) as float) as num_type_suite_Family
                      , cast(sum(case when name_type_suite in ('Group of people','Other_A','Other_B') then 1 else 0 end) as float) as num_type_suite_Other
                      , cast(sum(case when name_type_suite = 'Spouse, partner' then 1 else 0 end) as float) as num_type_suite_spouse
                      , cast(sum(case when name_type_suite = 'Unaccompanied' then 1 else 0 end) as float) as num_type_suite_Unaccompanied
                      
                      , cast(sum(case when name_client_type = 'New' then 1 else 0 end) as float) as num_client_type_New
                      , cast(sum(case when name_client_type = 'Refreshed' then 1 else 0 end) as float) as num_client_type_Refreshed
                      , cast(sum(case when name_client_type = 'Repeater' then 1 else 0 end) as float) as num_client_type_Repeater
                      , cast(sum(case when name_client_type = 'XNA' then 1 else 0 end) as float) as num_client_type_XNA
                      
                      , cast(sum(case when name_portfolio = 'Cards' then 1 else 0 end) as float) as num_portfolio_Cards
                      , cast(sum(case when name_portfolio = 'Cars' then 1 else 0 end) as float) as num_portfolio_Cars
                      , cast(sum(case when name_portfolio = 'Cash' then 1 else 0 end) as float) as num_portfolio_Cash
                      , cast(sum(case when name_portfolio = 'POS' then 1 else 0 end) as float) as num_portfolio_POS
                      , cast(sum(case when name_portfolio = 'XNA' then 1 else 0 end) as float) as num_portfolio_XNA
                      
                      , cast(sum(case when name_product_type = 'XNA' then 1 else 0 end) as float) as num_product_type_XNA
                      , cast(sum(case when name_product_type = 'walk-in' then 1 else 0 end) as float) as num_product_type_Walk_in
                      , cast(sum(case when name_product_type = 'x-sell' then 1 else 0 end) as float) as num_product_type_X_Sell
       
                      , cast(sum(case when channel_type = 'AP+ (Cash loan)' then 1 else 0 end) as float) as num_channel_type_AP
                      , cast(sum(case when channel_type = 'Car dealer' then 1 else 0 end) as float) as num_channel_type_Car_dealer
                      , cast(sum(case when channel_type = 'Channel of corporate sales' then 1 else 0 end) as float) as num_channel_type_Corp_sales
                      , cast(sum(case when channel_type = 'Contact center' then 1 else 0 end) as float) as num_channel_type_Contact_center
                      , cast(sum(case when channel_type = 'Country-wide' then 1 else 0 end) as float) as num_channel_type_country_wide
                      , cast(sum(case when channel_type = 'Credit and cash offices' then 1 else 0 end) as float) as num_channel_type_Credit_and_cash_offices
                      , cast(sum(case when channel_type = 'Regional / Local' then 1 else 0 end) as float) as num_channel_type_Regional_Local
                      , cast(sum(case when channel_type = 'Stone' then 1 else 0 end) as float) as num_channel_type_Stone
       
                      , cast(max(sellerplace_area) as float) as max_sellerplace_area
    
                      , cast(sum(case when name_yield_group = 'high' then 1 else 0 end) as float) as num_yield_group_high
                      , cast(sum(case when name_yield_group = 'low_action' then 1 else 0 end) as float) as num_yield_group_low_action
                      , cast(sum(case when name_yield_group = 'low_normal' then 1 else 0 end) as float) as num_yield_group_low_normal
                      , cast(sum(case when name_yield_group = 'middle' then 1 else 0 end) as float) as num_yield_group_middle
                      , cast(sum(case when name_yield_group = 'XNA' then 1 else 0 end) as float) as num_yield_group_XNA
    
                      , cast(avg(days_first_due*-1) as float) as days_first_due_avg_prev
                      , cast(max(days_first_due*-1) as float) as days_first_due_max_prev
                      , cast(min(days_first_due*-1) as float) as days_first_due_min_prev
    
                      , cast(avg(days_last_due_1st_version*-1) as float) as days_last_due_1st_version_avg_prev
                      , cast(max(days_last_due_1st_version*-1) as float) as days_last_due_1st_version_max_prev
                      , cast(min(days_last_due_1st_version*-1) as float) as days_last_due_1st_version_min_prev
    
                      , cast(avg(days_last_due*-1) as float) as days_last_due_avg_prev
                      , cast(max(days_last_due*-1) as float) as days_last_due_max_prev
                      , cast(min(days_last_due*-1) as float) as days_last_due_min_prev
                      
                      , cast(avg(days_first_drawing*-1) as float) as days_first_drawing_avg_prev
                      , cast(max(days_first_drawing*-1) as float) as days_first_drawing_max_prev
                      , cast(min(days_first_drawing*-1) as float) as days_first_drawing_min_prev
                      
                      , cast(avg(days_termination*-1) as float) as days_termination_avg_prev
                      , cast(max(days_termination*-1) as float) as days_termination_max_prev
                      , cast(min(days_termination*-1) as float) as days_termination_min_prev
                      
                      , cast(sum(nflag_insured_on_approval) as float) as num_insured
    
                      --continue
                      
                      
                      from raw_in.previous_application
                      group by sk_id_curr
                   ) as prev on bse.sk_id_curr = prev.sk_id_curr
        
        ;
                
        ALTER TABLE ABT.ABT_Full ADD PRIMARY KEY (sk_id_curr);
        """.format(
        amt_annuity_reg = amt_annuity_reg_str,
        amt_goods_price_reg = amt_goods_price_reg_str,

        )

        )

    conn.close()
