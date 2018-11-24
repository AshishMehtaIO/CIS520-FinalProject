function  pred_labels=lr_stepwise(train_inputs,train_labels,test_inputs)

    mns = mean(train_inputs,1);
    stds = std(train_inputs,1);
    x = (train_inputs - mns)./stds;
    
    %x = train_inputs;
    %x = [x ones(size(x,1),1)];
    
    % stepwi~ automatically includes bias term ones
    %ws
    [w1,~,~,inmodel1,~,~,~] = stepwisefit(x,train_labels(:,1));
    [w2,~,~,inmodel2,~,~,~] = stepwisefit(x,train_labels(:,2));
    [w3,~,~,inmodel3,~,~,~] = stepwisefit(x,train_labels(:,3));
    [w4,~,~,inmodel4,~,~,~] = stepwisefit(x,train_labels(:,4));
    [w5,~,~,inmodel5,~,~,~] = stepwisefit(x,train_labels(:,5));
    [w6,~,~,inmodel6,~,~,~] = stepwisefit(x,train_labels(:,6));
    [w7,~,~,inmodel7,~,~,~] = stepwisefit(x,train_labels(:,7));
    [w8,~,~,inmodel8,~,~,~] = stepwisefit(x,train_labels(:,8));
    [w9,~,~,inmodel9,~,~,~] = stepwisefit(x,train_labels(:,9));

    % training error    
    yh_1 = x * w1;
    yh_2 = x * w2;
    yh_3 = x * w3;
    yh_4 = x * w4;
    yh_5 = x * w5;
    yh_6 = x * w6;
    yh_7 = x * w7;
    yh_8 = x * w8;
    yh_9 = x * w9;
    
    % training error
    train_pred_labels = [yh_1 yh_2 yh_3 yh_4 yh_5 yh_6 yh_7 yh_8 yh_9];
    train_error = error_metric(train_pred_labels, train_labels);
    fprintf('stepwise trainig_error is %f \n',train_error);

    mns = mean(test_inputs,1);
    stds = std(test_inputs,1);
    xt = (test_inputs - mns)./stds;
    
    % training error    
    y_1 = xt * w1;
    y_2 = xt * w2;
    y_3 = xt * w3;
    y_4 = xt * w4;
    y_5 = xt * w5;
    y_6 = xt * w6;
    y_7 = xt * w7;
    y_8 = xt * w8;
    y_9 = xt * w9;
    
    pred_labels = [y_1 y_2 y_3 y_4 y_5 y_6 y_7 y_8 y_9];
end