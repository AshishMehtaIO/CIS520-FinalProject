function pred_labels = pls_regress (train_inputs, train_labels, test_inputs, d)
    % Principle Component Regression
    
    X = train_inputs;
    X_test = test_inputs;
    
    
    [N , P] = size(X);
    N_t = size(X_test, 1);
    Od = size(train_labels, 2);
         
    [PCALoadings,PCAScores, ~] = pca(X);
    
    mu = mean(X, 1);
 
    X_pca = (X - mu) * PCALoadings(:, 1:d);
    [X_pca, mu_1, sigma] = zscore(X_pca);
    
    X_test_pca = (X_test - mu)*PCALoadings(:, 1:d);
    X_test_pca = (X_test_pca - mu_1)./sigma;
    
    
    %weights = zeros(d+1, Od);
    
    
    y = train_labels;
    y1_std = (y - mean(y,1))./std(y,1);
    [XL,yl,XS,YS,beta,PCTVAR] = plsregress(X_pca,y1_std,15);

    %weights(:, i) = beta(:, 1);
    
    weights = beta;

    
        
%     for i = 1:Od
%         y = train_labels(:,i);
%         y1_std = (y - mean(y))./std(y);
%         [XL,yl,XS,YS,beta,PCTVAR] = plsregress(X_pca,y1_std,1);
%         
%         weights(:, i) = beta(:, 1);
% 
%     end
    
    mu = mean(train_labels,1);
    sigma = std(train_labels,1);
    
    X_pca = [ones(N, 1) X_pca];
    train_pred = X_pca * weights;
    train_pred = train_pred.*sigma + mu;
    train_error = error_metric(train_pred, train_labels);
    fprintf('Training error: %f\n',train_error)
    
    X_test_pca = [ones(N_t, 1) X_test_pca];
    pred_labels = X_test_pca * weights;
    
    pred_labels = pred_labels.* sigma + mu;
end