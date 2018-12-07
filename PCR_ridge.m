function pred_labels = PCR_ridge(train_inputs, train_labels, test_inputs, dim, penalty)
    % Principle Component Regression
    
    X = train_inputs;
    [PCALoadings,PCAScores,PCAVar] = pca(X);

    weights = zeros(size(X, 2)+1, size(train_labels, 2));
        
    for i = 1:size(train_labels, 2)
        y = train_labels(:,i);
        % train
        %betaPCR = regress(y-mean(y), PCAScores(:,1:d));
        betaPCR = ridge(y-mean(y), PCAScores(:,1:dim), penalty);
        betaPCR = PCALoadings(:,1:dim) * betaPCR;
        betaPCR = [mean(y) - mean(X) * betaPCR; betaPCR];
        weights(:,i) = betaPCR;
    end
    
    % predict
    train_pred = [ones(size(X, 1),1) X]*weights;
    train_error = error_metric(train_pred, train_labels);
    fprintf('Training error: %f\n',train_error);
 
    Xtest = test_inputs;
    pred_labels = [ones(size(Xtest, 1),1) Xtest] * weights;
end