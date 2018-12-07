function pred_labels = NCA (train_inputs, train_labels, test_inputs)
    % Nearest Component Analysis
    lambda = 1e-2;
    train_pred = [];    
    fprintf('TrainingModel');
    model1 = fsrnca(train_inputs, train_labels(:, 1), 'Verbose',0, ...
        'IterationLimit',30, 'Lambda', lambda);
    fprintf('TrainingModel');
    model2 = fsrnca(train_inputs, train_labels(:, 2), 'Verbose',0, ...
            'IterationLimit',30, 'Lambda', lambda);
    fprintf('TrainingModel');    
    model3 = fsrnca(train_inputs, train_labels(:, 3), 'Verbose',0, ...
            'IterationLimit',30, 'Lambda', lambda);
        fprintf('TrainingModel');
    model4 = fsrnca(train_inputs, train_labels(:, 4), 'Verbose',0, ...
            'IterationLimit',30, 'Lambda', lambda);
        fprintf('TrainingModel');
    model5 = fsrnca(train_inputs, train_labels(:, 5), 'Verbose',0, ...
            'IterationLimit',30, 'Lambda', lambda);
        fprintf('TrainingModel');
    model6 = fsrnca(train_inputs, train_labels(:, 6), 'Verbose',0, ...
            'IterationLimit',30, 'Lambda', lambda);
        fprintf('TrainingModel');
    model7 = fsrnca(train_inputs, train_labels(:, 7), 'Verbose',0, ...
            'IterationLimit',30, 'Lambda', lambda);
        fprintf('TrainingModel');
    model8 = fsrnca(train_inputs, train_labels(:, 8), 'Verbose',0, ...
            'IterationLimit',30, 'Lambda', lambda);
        fprintf('TrainingModel');
    model9 = fsrnca(train_inputs, train_labels(:, 9), 'Verbose',0, ...
            'IterationLimit',30, 'Lambda', lambda);    
        
    train_pred1 = predict(model1,train_inputs);
    train_pred2 = predict(model2,train_inputs);
    train_pred3 = predict(model3,train_inputs);
    train_pred4 = predict(model4,train_inputs);
    train_pred5 = predict(model5,train_inputs);
    train_pred6 = predict(model6,train_inputs);
    train_pred7 = predict(model7,train_inputs);
    train_pred8 = predict(model8,train_inputs);
    train_pred9 = predict(model9,train_inputs);
    
    train_pred = [train_pred1 ,train_pred2, train_pred3, train_pred4, ...
        train_pred5, train_pred6, train_pred7, train_pred8, train_pred9];
    
    train_error = error_metric(train_pred, train_labels);
    fprintf('Training error: %f\n',train_error);
    
    test_pred1 = predict(model1,test_inputs);
    test_pred2 = predict(model2,test_inputs);
    test_pred3 = predict(model3,test_inputs);
    test_pred4 = predict(model4,test_inputs);
    test_pred5 = predict(model5,test_inputs);
    test_pred6 = predict(model6,test_inputs);
    test_pred7 = predict(model7,test_inputs);
    test_pred8 = predict(model8,test_inputs);
    test_pred9 = predict(model9,test_inputs);
    
    pred_labels = [test_pred1 ,test_pred2, test_pred3, test_pred4, ...
        test_pred5, test_pred6, test_pred7, test_pred8, test_pred9];
end