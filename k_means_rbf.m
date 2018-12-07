function pred_labels = k_means_rbf(train_inputs, train_labels, test_inputs, k, sig)

    train_inputs = (train_inputs - mean(train_inputs,1))./std(train_inputs,1);
    test_inputs = (test_inputs - mean(train_inputs,1))./std(train_inputs,1); 

%     X= train_inputs;
%     d=150;
%     [PCALoadings,PCAScores,PCAVar] = pca(X);
%     train_inputs = PCAScores(:,1:d);
%     test_inputs = (test_inputs - mean(X)) * PCALoadings(:,1:d);
    
    [idx,c] =kmeans(train_inputs,k);
    
    n = size(train_inputs,1);
    
    phi_k = zeros(n,k);
    
    for i=1:n
       X_t = train_inputs(i,:);
       X_t_rep = repmat(X_t,k,1);
       dist = sum((X_t_rep - c).^2,2);
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
       X_t = test_inputs(i,:);
       X_t_rep = repmat(X_t,k,1);
       dist = sum((X_t_rep - c).^2,2);
       phi(i,:) = (exp(-dist./sig))';
    end
    
    % predict
    train_pred = [ones(size(phi_k, 1),1) phi_k]*weights;
    train_error = error_metric(train_pred, train_labels);
%     fprintf('Training error: %f\n',train_error);
     
    pred_labels = [ones(size(phi, 1),1) phi] * weights;
%   fprintf('end');
    
end