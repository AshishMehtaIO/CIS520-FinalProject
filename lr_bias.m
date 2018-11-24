function  pred_labels=lr_bias(train_inputs,train_labels,test_inputs)

%     % standardize_X
    mns = mean(train_inputs,1);
    stds = std(train_inputs,1);
    xstd = (train_inputs - mns)./stds;
    
    %xstd = train_inputs;
    xstd = [xstd ones(size(xstd,1),1)];
    
    %inverse = inv(x'*x)*x';
    inverse = pinv(xstd);
    
    w = inverse * train_labels;
    
    % training error
        
    train_pred_labels = xstd * w;
    train_error = error_metric(train_pred_labels, train_labels);
    fprintf('trainig_error std is %f \n',train_error);

%     % standardize_Xtest
    mns = mean(test_inputs,1);
    stds = std(test_inputs,1);
    xtstd = (test_inputs - mns)./stds;
    %xtstd = test_inputs;
    
    pred_labels = [xtstd ones(size(xtstd,1),1)] * w; 

end