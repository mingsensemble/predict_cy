# Predict Cy Young Winners

This small machine learning project predicts Cy Young Award winners by stacking multiple models: neural network, random forest, gradient boosted trees, and support vector machines.

## Data Source
The model is developed on data from 2006 to 2017.  Historical Cy Young award winner records are obtained from ESPN websites, and the performance statistics are downloaded from Fangraphs.  2006 season is used as the initial season because Fangraphs started to record advanced statistics since 2006.

## Programs
The development scripts are organized with a four-digit number to indicate the order of running the codes.  

* __1000-preprocess_data__: download data from Fangraphs and merge with Cy Young winner data
* __2000-run_model_selection_on_aws__: run stratefied cross validation by season; use Brier score as the performance metric.  The script is designed to be run on aws
* __3000-run_keras_models__: use Keras to develop a neural network model.  The models are tuned on a limited parameter space
* __4000-model_stacking__: use logistic regression to identify the optimal weights for each algorithm for ensembling
* __predict_cy_production_2018__: the previous scripts are combined and streamlined to create this production script where `python3 predict_cy_production_2018.py` generates an csv file that shows the final prediction

## Results
The prediction results for 2018 season as fo the execution date can be found at __./Results/stack_results.csv__.