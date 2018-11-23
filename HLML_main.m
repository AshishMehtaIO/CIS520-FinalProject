function HLML_main()

%% Params
validation_percentage = 25;
num_folds = 10;
%% main

[val_input, val_labels, train_cval_input, train_cval_labels, ...
    train_cval_parts] = load_data(validation_percentage, num_folds);

for iter_num = 1:max(train_cval_parts)
    
    fprintf('\n\nCross validation iteration %d\n\n', iter_num);
    
    [Xtrain, Ytrain, XCV, YCV] = make_folds(train_cval_parts, ...
        train_cval_input, train_cval_labels, iter_num);
  
    % model, training_error = some_training_fn(Xtrain, Ytain)
    % YCV_hat = @model(XCV)
    % cv_error = error_metric(YCV_hat, YCV)
end


end