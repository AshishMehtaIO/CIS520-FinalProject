function pred_labels = novel_PCR (train_inputs, train_labels, test_inputs)
    % Principle Component Regression
    d = 170;
    X = train_inputs(:, 22:end);
    [PCALoadings,PCAScores,PCAVar] = pca(X);

    weights = zeros(size(X, 2)+1, size(train_labels, 2));
        
    for i = 1:size(train_labels, 2)
        y = train_labels(:,i);
        % train
        betaPCR = regress(y-mean(y), PCAScores(:,1:d));
        betaPCR = PCALoadings(:,1:d) * betaPCR;
        betaPCR = [mean(y) - mean(X) * betaPCR; betaPCR];
        weights(:,i) = betaPCR;
    end
    
    % predict twitter
    train_pred_twitter = [ones(size(X, 1),1) X]*weights;
    train_error_twitter = error_metric(train_pred_twitter, train_labels);
    fprintf('Twitter Training error: %f\n',train_error_twitter);
 
    Xtest = test_inputs(:, 22:end);
    pred_labels_twitter = [ones(size(Xtest, 1),1) Xtest] * weights;
    
    % predict 
    X_demo = [ones(size(train_inputs,1),1) train_inputs(:, 1:21)];
    w = pinv(X_demo)*train_labels;
    
    pred_labels_demo = [ones(size(test_inputs, 1), 1) test_inputs(:, 1:21)] * w;
    
    pred_labels = (pred_labels_demo + pred_labels_twitter)./2;
end