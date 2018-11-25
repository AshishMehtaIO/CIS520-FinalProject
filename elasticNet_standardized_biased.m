function pred_labels = elasticNet_standardized_biased (train_inputs, ...
    train_labels,test_inputs)
    % Lasso regression with bias and standardization
    
    test_mean = mean(test_inputs, 1);
    test_std_dev = std(test_inputs, 1);
    std_test_inputs = (test_inputs - test_mean)./test_std_dev;
    
    train_mean = mean(train_inputs, 1);
    train_std_dev = std(train_inputs, 1);
    std_train_inputs = (train_inputs - train_mean)./train_std_dev;
    
    weights = zeros(size(train_inputs, 2), size(train_labels, 2));
    biases = zeros(1, size(train_labels, 2));
    
    for i = 1:size(train_labels, 2)
        
        fprintf('Dimension Iteration %d\n', i);
        [B,FitInfo] = lasso(train_inputs, train_labels(:, i),...
            'Alpha',0.005, 'MaxIter', 1e4, 'Lambda', 700);
    
        weights(:, i) = B;
        biases (i) = FitInfo.Intercept;
    
    end
    
    train_pred = std_train_inputs* weights + biases;
    train_error = error_metric(train_pred, train_labels);
    fprintf('Training error: %f\n',train_error);
    
    pred_labels = std_test_inputs * weights + biases;
end