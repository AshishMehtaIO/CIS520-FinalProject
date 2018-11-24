function pred_labels = ridge_standardized_biased (train_inputs, ...
    train_labels,test_inputs)
    % Ridge regression with bias and standardization 
    
    train_inputs_ones = [ones(size(train_inputs, 1), 1), train_inputs];
    test_inputs_ones = [ones(size(test_inputs, 1), 1), test_inputs];
    lambda = 700;
    weights = zeros(size(train_inputs_ones, 2), size(train_labels, 2));
    for j = 1:size(lambda, 2)
        fprintf("Lambda = %f\n", lambda(j));
        for i = 1:size(train_labels, 2)
            weights(:, i) = ridge(train_labels(:, i), train_inputs, ...
            lambda(j), 0);
        end
    end
    train_pred = train_inputs_ones * weights;
    train_error = error_metric(train_pred, train_labels);
    fprintf('Training error: %f\n',train_error);
    
    pred_labels = test_inputs_ones * weights;
    
end