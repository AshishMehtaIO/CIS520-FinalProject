function pred_labels = glm(train_inputs,train_labels,test_inputs, alpha, beta, gamma)
    
    train_inputs_ones = [ones(size(train_inputs, 1), 1), train_inputs];
    test_inputs_ones = [ones(size(test_inputs, 1), 1), test_inputs];

    weights = zeros(size(train_inputs_ones, 2), size(train_labels, 2));
 
    for i = 1:size(train_labels, 2)
        weights(:, i) = glmfit(train_inputs, train_labels(:, i), ...
            'poisson');
    end
    
    train_pred = train_inputs_ones * weights;
    train_error = error_metric(train_pred, train_labels);
    fprintf('Training error: %f\n',train_error);
    
    pred_labels = test_inputs_ones * weights;
    
end