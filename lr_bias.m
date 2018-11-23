function  pred_labels=lr_bias(train_inputs,train_labels,test_inputs)

    % standardize_X
    %     mns = mean(train_inputs,1);
    %     stds = std(train_inputs,1);
    %     x = (train_inputs - mns)./stds;
    
    x = train_inputs;
    x = [x ones(size(x,1),1)];
    
    %inverse = inv(x'*x)*x';
    inverse = pinv(x);
    
    w = inverse * train_labels;
    
    pred_labels = [test_inputs ones(size(test_inputs,1),1)] * w; 

end