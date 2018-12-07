function pred_labels = k_means_rbf_new2(train_inputs, train_labels, test_inputs, k, sig)
    %rng(500);
    X = train_inputs;
    X_test = test_inputs;
    
    N = size(X, 1);
    N_t = size(X_test, 1);
    Od = size(train_labels, 2);
         
    [PCALoadings,PCAScores, ~] = pca(X);
    
    mu = mean(X, 1);
 
    d = 144;
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
        y1_std = (y - mean(y))./std(y);
        %weights(:,i) = regress(y1_std,[ones(size(phi_k, 1),1) phi_k]);
        weights(:,i) = ridge(y1_std, phi_k, 0.05, 0);
        
%         y1_std = (y - mean(y))./std(y);
%         [XL,yl,XS,YS,beta,PCTVAR] = plsregress(phi_k,y1_std,50);
%         weights(:, i) = beta;
    end
        
%         y = train_labels;
%         y1_std = (y - mean(y,1))./std(y,1);
%         [XL,yl,XS,YS,beta,PCTVAR] = plsregress(phi_k,y1_std,70);
%         %weights(:, i) = beta;
%         weights = beta;
   
   nt = size(test_inputs,1);
   
   phi = zeros(nt,k);
    
    for i=1:nt
       X_t = X_test_pca(i,:);
       dist = sum((X_t - c).^2,2);
       phi(i,:) = (exp(-dist./sig))';
    end
    
    % predict
    mu = mean(train_labels,1);
    sigma = std(train_labels,1);
    
    train_pred = [ones(size(phi_k, 1),1) phi_k]*weights;
    train_pred = train_pred.*sigma + mu;
    train_error = error_metric(train_pred, train_labels);
    fprintf('Training error: %f\n',train_error);
     
    pred_labels = [ones(size(phi, 1),1) phi] * weights;
    
    pred_labels = pred_labels.*sigma + mu;
    fprintf('end');
    
end