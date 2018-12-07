function pred_labels = k_means_rbf_new(train_inputs, train_labels, test_inputs, k, sig)

    rng(500);

    X = train_inputs;
    X_test = test_inputs;
    
    N = size(X, 1);
    N_t = size(X_test, 1);
    Od = size(train_labels, 2);
         
    [PCALoadings,PCAScores, ~] = pca(X);
    mu = mean(X, 1);
 
    d = 145;
    X_pca = (X - mu) * PCALoadings(:, 1:d);
    [X_pca, mu_1, sigma] = zscore(X_pca);
    %X_pca = [ones(N, 1) X_pca];
    
    X_test_pca = (X_test - mu)*PCALoadings(:, 1:d);
    X_test_pca = (X_test_pca - mu_1)./sigma;


    [idx,c] =kmeans(X_pca,k);
    
    n = size(train_inputs,1);
    
    phi_k = zeros(n,k);
    
    for i=1:n
       X_t = X_pca(i,:);
       dist = sum((X_t - c).^2,2);
       phi_k(i,:) = (exp(-dist./sig))';
    end
    
%     fprintf('%d',size(phi_k));
    %  kx9     
    weights = zeros(size(phi_k,2)+1,size(train_labels,2));
    
    for i = 1:size(train_labels, 2)
        y = train_labels(:,i);    
        weights(:,i) = regress(y,[ones(size(phi_k, 1),1) phi_k]);
    end
   
   nt = size(test_inputs,1);
   
   phi = zeros(nt,k);
    
    for i=1:nt
       X_t = X_test_pca(i,:);
       dist = sum((X_t - c).^2,2);
       phi(i,:) = (exp(-dist./sig))';
    end
    
    % predict
    train_pred = [ones(size(phi_k, 1),1) phi_k]*weights;
    train_error = error_metric(train_pred, train_labels);
    fprintf('Training error: %f\n',train_error);
     
    pred_labels = [ones(size(phi, 1),1) phi] * weights;
    fprintf('end');
    
end