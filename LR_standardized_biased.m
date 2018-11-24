function pred_labels = LR_standardized_biased (train_inputs, ...
    train_labels,test_inputs)
    % Function to first standardize the dataset and run Linear Regression
    
    train_inputs_ones = [ones(size(train_inputs, 1), 1), train_inputs];
    test_inputs_ones = [ones(size(test_inputs, 1), 1), test_inputs];
    weights = zeros(size(train_inputs_ones, 2), size(train_labels, 2));
    for i = 1:size(train_labels, 2)
        weights(:, i) = regress(train_labels(:, i), train_inputs_ones);
    end
    
    train_pred = train_inputs_ones * weights;
    train_error = error_metric(train_pred, train_labels);
    fprintf('Training error: %f\n',train_error);
    
    pred_labels = test_inputs_ones * weights;
    
end