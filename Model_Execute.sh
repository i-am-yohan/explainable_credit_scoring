

#Create directory to put all of the Kaggle competitions
mkdir Kaggle_Submissions

#installing required packages
pip3 install -r Scripts/Requirements.txt

#Execute model scripts
python3 Scripts/01_Data_Cleaning_and_Feature_Engineering.py $1 $2
python3 Scripts/02_Sampling.py $1 $2
Rscript Scripts/03_1_Scorecard_Dev.R $1 $2
python3 Scripts/03_2_RandomForest_Dev.py $1 $2
python3 Scripts/03_3_LightGBM_Dev.py $1 $2
python3 Scripts/03_4_XGBoost_Dev.py $1 $2

#Submit Results
~/.local/bin/kaggle competitions submit -c home-credit-default-risk -f Kaggle_Submissions/01_Scorecard.csv -m 01_Credit_Scorecard_Depl_Sub

~/.local/bin/kaggle competitions submit -c home-credit-default-risk -f Kaggle_Submissions/02_Random_Forest.csv -m 02_Random_Forest_Depl_Sub

~/.local/bin/kaggle competitions submit -c home-credit-default-risk -f Kaggle_Submissions/03_LightGBM.csv -m 03_LightGBM_Depl_Sub

~/.local/bin/kaggle competitions submit -c home-credit-default-risk -f Kaggle_Submissions/04_XGBoost.csv -m 04_XGBoost_Depl_Sub

#Create tables for visualization
python3 Scripts/04_Expl_Data_Create.py $1 $2
