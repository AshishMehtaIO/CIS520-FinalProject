function pred_labels = predict_labels(train_inputs,train_labels,test_inputs)

    % Function that has to be modified with your training function
    % Modify the function name below as required. An example
    % random_precitor baseline is shown below

    % pred_labels = your_training_function(train_inputs, train_labels,
    % test_inputs);
    % pred_labels = random_predictor(train_inputs,train_labels,test_inputs, ...
    %    0, 0, 0);
    
    % nvaive_linear_regression (may fail if xTx is not inveritible)
%     pred_labels = lr_bias(train_inputs,train_labels,test_inputs);
    % pred_labels = LR_combined(train_inputs,train_labels,test_inputs,150);
    
    % pred_labels = standardized_LR(train_inputs,train_labels,test_inputs);
    % pred_labels = lr_stepwise(train_inputs,train_labels,test_inputs);

    % pred_labels = elasticNet_standardized_biased(train_inputs, ...
    % train_labels,test_inputs);

    %pred_labels = PCR(train_inputs, train_labels, test_inputs, 144);
    % pred_labels = PCR_ridge(train_inputs, train_labels, test_inputs, 90);
    % pred_labels = glm(train_inputs,train_labels,test_inputs, 0, 0, 0);
    %pred_labels = RandomForest_reg(train_inputs, train_labels, test_inputs);
    % pred_labels = pls_regress(train_inputs, train_labels, test_inputs, 144);
    % pred_labels =  k_means_rbf(train_inputs, train_labels, test_inputs, 400, 300);
    %pred_labels = LR_standardized_biased (train_inputs,train_labels,test_inputs);
%     lambda = 50;
% 	pred_labels = ridge_standardized_biased(train_inputs,train_labels,test_inputs);
    k = 250;
    sig = 1500;
    pred_labels = k_means_rbf_new2(train_inputs, train_labels, test_inputs, k, sig);
    
end
