function pred_labels = standardized_LR (train_inputs, ...
    train_labels,test_inputs)
    % Function to first standardize the dataset and run Linear Regression
    
    train_inputs_ones = [ones(size(train_inputs, 1), 1), train_inputs];
    test_inputs_ones = [ones(size(test_inputs, 1), 1), test_inputs];
    weights = mvregress(train_inputs_ones, train_labels);
    
    pred_labels = test_inputs_ones * weights;   
end