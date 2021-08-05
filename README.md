# explainable_credit_scoring
The code for the "Improving Credit Default Prediction with Explainable AI" masters project.

Please see the following step-by-step process to run the project.

## Step 1: Build the database required for execution.
Follow the instructions in the following link:
https://github.com/i-am-yohan/home_credit_2_postgres
This will download all of the required files and upload them to a postgres database. This is not the most efficient method of doing it so feel free to only use this as a guide.

## Step 2: Executing the models and submitting to Kaggle.
Execute the `Model_Execute.sh` to build the models and make the Kaggle submissions. This is done as follows:
```
./Model_Execute.sh <postgres id> <postgres password>
```
Ensure this is executed in the same folder as the Scripts sub-folder.

## Step 3: Extracting the counterfactuals.
The following inputs to the counterfactual search alogorithm are as follows:
1. `sk_id_curr` - The observation of interest.
2. `target_score` - The target counterfactual score
3. `n_samples` - The number of samples to take at each iteration

The script is executed as follows:
```
python3 Scripts/05_Extract_CF.py <postgres id> <postgres password> <sk_id_curr> <target_score> <n_samples>
```
This loads a table to the postgresDB called `plot_table`. This contains the counterfactual information.

## Step 4: Visualize the counterfactuals
The `06_Visualization.py` script uses the `plot_table` to create graphs to help visualize the project. It is executed as follows:
```
python3 Scripts/06_Visualization.py <postgres id> <postgres password>
```
These are output as plotly graphs sent to the web browser.
