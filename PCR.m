function pred_labels = PCR (train_inputs, train_labels, test_inputs)
    % Principle Component Regression
    
    mns = mean(train_inputs,1);
    stds = std(train_inputs,1);
<<<<<<< HEAD
    X = train_inputs;
    %X = (train_inputs - mns)./stds;
=======
    X = (train_inputs - mns)./stds;
>>>>>>> b41d77398d7c8735c8df28f84e54dfc74b10cb49
    
    [PCALoadings,PCAScores,PCAVar] = pca(X);

    weights = zeros(size(X, 2)+1, size(train_labels, 2));

<<<<<<< HEAD
    
    d = 92;
    
    for i = 1:size(train_labels, 2)
        y = train_labels(:,i);
        % train
        betaPCR = regress(y-mean(y), PCAScores(:,1:d));
        betaPCR = PCALoadings(:,1:d) * betaPCR;
=======
    for i = 1:size(train_labels, 2)
        
        y = train_labels(:,i);
        % train
        betaPCR = regress(y-mean(y), PCAScores(:,1:2));
        betaPCR = PCALoadings(:,1:2) * betaPCR;
>>>>>>> b41d77398d7c8735c8df28f84e54dfc74b10cb49
        betaPCR = [mean(y) - mean(X) * betaPCR; betaPCR];
        weights(:,i) = betaPCR;
    end
    
    % predict
    train_pred = [ones(size(X, 1),1) X]*weights;
    train_error = error_metric(train_pred, train_labels);
    fprintf('Training error: %f\n',train_error);
 
    mns = mean(test_inputs,1);
    stds = std(test_inputs,1);
<<<<<<< HEAD
    Xtest = test_inputs;
    %Xtest = (test_inputs - mns)./stds;
=======
    Xtest = (test_inputs - mns)./stds;
>>>>>>> b41d77398d7c8735c8df28f84e54dfc74b10cb49
    
    pred_labels = [ones(size(Xtest, 1),1) Xtest] * weights;
end