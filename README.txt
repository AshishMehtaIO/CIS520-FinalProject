Experimental observations & inferences:
-LR:
	-inv(X'X) is close to singluar. output predictions were negative. val_error ~ 90.
	-inv((1,x)' (1,x)) (bias added) is better but still val_error ~ 60.
	-pinv((1,x)) provides gives val_error ~ 0.12. but training error = 0. Indicates overfitting. Regularization penalty should be used.
	-pinv((1,x)) with standardization gives training error ~ 1.5 and val_error ~ 1.5. (does not overfit but performs poorly).
	-using regress with standardization and biased gives 0 training error and 0.68 validation error.
	
-Ridge Regression:
	-with lambda = 1e-2, training_error ~ 0.000008, avg_cv = 0.127
	-with lambda = 1.5e-2, training_error ~ 0.000012, avg_cv = 0.128						
	-with lambda = 1.5e-1, training_error ~ 0.000119, avg_cv = 0.126
	-with lambda = 1, training_error ~ 0.000781, avg_cv = 0.125
	-with lambda = 100, training_error ~ 0.02, avg_cv = 0.093256
	-with lambda = 10, training_error ~ 0.005, avg_cv = 0.116140 (higher lambda required)
	-with lambda = 1000, training_error ~ 0.05, avg_cv = 0.086762
	-with lambda = 10000, training_error ~ 0.09, avg_cv = 0.104666 (ideal lambda between 100 and 10000)
	-with lambda = 3000, training_error ~ 0.075, avg_cv = 0.09352 (ideal lambda between 100 and 3000)
	-with lambda = 700, training_error ~ 0.05, avg_cv = 0.085831 (ideal lambda between 100 and 1000)
	-with lambda = 300, training_error ~ 0.037, avg_cv = 0.086693 (ideal lambda between 300 and 1000)
	
	-validation error: 0.090769, test error: 0.0872.

-Stepwise
	-selecting ~300 for each y, without_standardizing the data, tr_err = 3.314. val = 3.319	

	
-PCR
	- Reduced Dim = 90
	- Performance: training error = 0.068341, validation error =0.082862 , test error = 0.0802 .
	
	- PCR with ridge - 0.14 best after cval over a range ...

-Elastic Net




-GMM/KMeans
-RBF
-Kernel Regression/KNN
-SVM regression
-Generalized Linear Models
-PCA/autoencoder
-gradient boosting(for ensemble)				
-split input dataset
-NN (maybe)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
training_data.csv

Features and labels are contained in the file training_data.csv. Here is the description of each column in the file.

FIPS Code: Column 1 containts the unique FIPS code of each county. Each row is the data of one county, and there are 1019 counties.

Features: Columns 2-22 are demographic and SES features, columns 23-2022 are LDA topic frequencies from tweets.
Labels: Columns 2123-2031 are health outcomes that you need to predict.

-----------------------------------------------------------------------

Demographic and SES features are:

['demo_pcblack', 'demo_pcfemale', 'demo_pchisp', 'demo_pcwht', 'demo_under18', 'demo_65over', 'largemetro', 'mediummetro', 'metrononmetro', 'micro', 'nchs_2013', 'ses_edu_coll', 'ses_foodenvt', 'ses_incomeratio', 'ses_log_hhinc', 'ses_pcaccess', 'ses_pcexerciss', 'ses_pchousing', 'ses_pcrural', 'ses_pcunemp', 'smallmetro']


LDA topic frequencies:
These columns contain frequency of the topic in that county. First row contains the topic ID, and each topic can be better understood by looking at the top 20 frequent words of that topic in the topics.csv file. 


Health Outcomes  are:

['health_aamort', 'health_fairpoor', 'health_mentunh', 'health_pcdiab', 'health_pcexcdrin', 'health_pcinact', 'health_pcsmoker', 'health_physunh', 'heath_pcobese']

-----------------------------------------------------------------------
-----------------------------------------------------------------------
-----------------------------------------------------------------------

training_data.mat

This file contains the same data as training_data.csv, but split into two matlab matrices named training_inputs and training_labels. Columns in matrices have the same order as in the csv file.

-----------------------------------------------------------------------
-----------------------------------------------------------------------
-----------------------------------------------------------------------
topics.csv

This file contains top 20 words for each topic. The topic IDs are in the first column and they correspond to the LDA topic frequencies columns in the training_data.csv file.
Each row is one topic, with topic ID in first column followed by top 20 most frequent words and their frequencies.


-----------------------------------------------------------------------
-----------------------------------------------------------------------
-----------------------------------------------------------------------

error_metric.m

This file contains the code for error metric that will be used for grading

-----------------------------------------------------------------------
-----------------------------------------------------------------------
-----------------------------------------------------------------------

predict_labels.m

This file contains a sample submission
