# explainable_credit_scoring
The code for the "Improving Credit Default Prediction with Explainable AI" masters project.

Please see the following step-by-step process to run the project.
Step 1: Build the database required for execution.
Follow the instructions in the following link:
https://github.com/i-am-yohan/home_credit_2_postgres
This will download all of the required files and upload them to a postgres database. This is not the most efficient method of doing it so feel free to only use this as a guide.

Step 2: Executing the models and submitting to Kaggle.
Execute the `Model_Execute.sh` to build the models and make the Kaggle submissions. This is done as follows:
```
./Model_Execute.sh <postgres id> <postgres password>
```

Step 3: Extracting the counterfactuals.
