function pred_labels = RandomForest_reg(train_inputs, train_labels, test_inputs)
    % Decision tree random forest regression
        
    X= train_inputs;
    d=150;
    [PCALoadings,PCAScores,PCAVar] = pca(X);
    train_inputs = PCAScores(:,1:d);

    test_inputs = (test_inputs - mean(X)) * PCALoadings(:,1:d);   
    
    MDL_1 = fitrensemble(train_inputs,train_labels(:,1),'LearnRate',0.1,'Learners',templateTree('MaxNumSplits',20),'NumLearningCycles',10);
    MDL_2 = fitrensemble(train_inputs,train_labels(:,2),'LearnRate',0.1,'Learners',templateTree('MaxNumSplits',20),'NumLearningCycles',10);
    MDL_3 = fitrensemble(train_inputs,train_labels(:,3),'LearnRate',0.1,'Learners',templateTree('MaxNumSplits',20),'NumLearningCycles',10);
    MDL_4 = fitrensemble(train_inputs,train_labels(:,4),'LearnRate',0.1,'Learners',templateTree('MaxNumSplits',20),'NumLearningCycles',10);
    MDL_5 = fitrensemble(train_inputs,train_labels(:,5),'LearnRate',0.1,'Learners',templateTree('MaxNumSplits',20),'NumLearningCycles',10);
    MDL_6 = fitrensemble(train_inputs,train_labels(:,6),'LearnRate',0.1,'Learners',templateTree('MaxNumSplits',20),'NumLearningCycles',10);
    MDL_7 = fitrensemble(train_inputs,train_labels(:,7),'LearnRate',0.1,'Learners',templateTree('MaxNumSplits',20),'NumLearningCycles',10);
    MDL_8 = fitrensemble(train_inputs,train_labels(:,8),'LearnRate',0.1,'Learners',templateTree('MaxNumSplits',20),'NumLearningCycles',10);
    MDL_9 = fitrensemble(train_inputs,train_labels(:,9),'LearnRate',0.1,'Learners',templateTree('MaxNumSplits',20),'NumLearningCycles',10);
    fprintf('built model \n');    
    
    % predict
    train_pred_1 = predict(MDL_1,train_inputs);
    fprintf('predicted classes \n');
    train_pred_2 = predict(MDL_2,train_inputs);
    train_pred_3 = predict(MDL_3,train_inputs);
    train_pred_4 = predict(MDL_4,train_inputs);
    train_pred_5 = predict(MDL_5,train_inputs);
    train_pred_6 = predict(MDL_6,train_inputs);
    train_pred_7 = predict(MDL_7,train_inputs);
    train_pred_8 = predict(MDL_8,train_inputs);
    train_pred_9 = predict(MDL_9,train_inputs);

    train_pred = [train_pred_1 train_pred_2 train_pred_3 train_pred_4 train_pred_5 train_pred_6 train_pred_7 train_pred_8 train_pred_9];
    train_error = error_metric(train_pred, train_labels);
    fprintf('Training error: %f\n',train_error);

    % predict
    fprintf('predicting test classes');
    test_pred_1 = predict(MDL_1,test_inputs);
    test_pred_2 = predict(MDL_2,test_inputs);
    test_pred_3 = predict(MDL_3,test_inputs);
    test_pred_4 = predict(MDL_4,test_inputs);
    test_pred_5 = predict(MDL_5,test_inputs);
    test_pred_6 = predict(MDL_6,test_inputs);
    test_pred_7 = predict(MDL_7,test_inputs);
    test_pred_8 = predict(MDL_8,test_inputs);
    test_pred_9 = predict(MDL_9,test_inputs);

    pred_labels = [test_pred_1 test_pred_2 test_pred_3 test_pred_4 test_pred_5 test_pred_6 test_pred_7 test_pred_8 test_pred_9];   
end