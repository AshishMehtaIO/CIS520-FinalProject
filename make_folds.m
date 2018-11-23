function [Xtrain, Ytrain, XCV, YCV] = ...
    make_folds(train_cval_parts, train_cval_input, train_cval_label, N)

    % Get the training data
    Xtrain = train_cval_input(train_cval_parts ~= N, :);
    Ytrain = train_cval_label(train_cval_parts ~= N, :);

    % Get the testing data
    XCV = train_cval_input(train_cval_parts == N, :);
    YCV = train_cval_label(train_cval_parts == N, :);
    
end