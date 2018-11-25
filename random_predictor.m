function pred_labels = random_predictor(train_inputs,train_labels,test_inputs, alpha, beta, gamma)
    
    pred_labels=randn(size(test_inputs,1),size(train_labels,2));
    
end
