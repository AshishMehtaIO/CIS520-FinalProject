function pred_labels = PCR (train_inputs, train_labels, test_inputs)
    % Principle Component Regression
    
    mns = mean(train_inputs,1);
    stds = std(train_inputs,1);
    X = train_inputs;
    %X = (train_inputs - mns)./stds;
    
    [PCALoadings,PCAScores,PCAVar] = pca(X);

    weights = zeros(size(X, 2)+1, size(train_labels, 2));

    
    d = 92;
    
    for i = 1:size(train_labels, 2)
        y = train_labels(:,i);
        % train
        betaPCR = regress(y-mean(y), PCAScores(:,1:d));
        betaPCR = PCALoadings(:,1:d) * betaPCR;
        betaPCR = [mean(y) - mean(X) * betaPCR; betaPCR];
        weights(:,i) = betaPCR;
    end
    
    % predict
    train_pred = [ones(size(X, 1),1) X]*weights;
    train_error = error_metric(train_pred, train_labels);
    fprintf('Training error: %f\n',train_error);
 
    mns = mean(test_inputs,1);
    stds = std(test_inputs,1);
    Xtest = test_inputs;
    %Xtest = (test_inputs - mns)./stds;
    
    pred_labels = [ones(size(Xtest, 1),1) Xtest] * weights;
end