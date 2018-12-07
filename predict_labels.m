function pred_labels = predict_labels(train_inputs,train_labels,test_inputs)

    % Function that has to be modified with your training function
    % Modify the function name below as required. An example
    % random_precitor baseline is shown below

    % pred_labels = your_training_function(train_inputs, train_labels,
    % test_inputs);
    % pred_labels = random_predictor(train_inputs,train_labels,test_inputs, ...
    %    0, 0, 0);
    
    % nvaive_linear_regression (may fail if xTx is not inveritible)
    % pred_labels = lr_bias(train_inputs,train_labels,test_inputs);
    
    % pred_labels = standardized_LR(train_inputs,train_labels,test_inputs);
    % pred_labels = lr_stepwise(train_inputs,train_labels,test_inputs);

%     pred_labels = elasticNet_standardized_biased(train_inputs, ...
%     train_labels,test_inputs);

    % pred_labels = glm(train_inputs,train_labels,test_inputs, 0, 0, 0);
%     pred_labels = kernelRegression_PCA(train_inputs,...
%         train_labels,test_inputs);
pred_labels = novel_PCR (train_inputs, ...
        train_labels, test_inputs);
end
