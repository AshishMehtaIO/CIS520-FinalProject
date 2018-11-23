function pred_labels = random_predictor(train_inputs,train_labels,test_inputs)
    
    pred_labels=randn(size(test_inputs,1),size(train_labels,2));
    
end
