function pred_labels = Ensemble(train_inputs, train_labels, test_inputs)
    % Decision tree random forest regression
%     
      d=144;
      [PCALoadings,PCAScores,PCAVar] = pca(train_inputs(:, 22:end));
%       train_inputs = PCAScores(:,1:d);

    m_train = mean(train_inputs(:, 22:end));
%     train_inputs(:, 22:end) = (train_inputs(:, 22:end) - m_train)./std(train_inputs(:, 22:end));
%     test_inputs(:, 22:end) = (test_inputs(:, 22:end) - m_train)./std(train_inputs(:, 22:end));
    train_inputs_pca = (train_inputs(:, 22:end) - m_train)* PCALoadings(:,1:d);   
    test_inputs_pca = (test_inputs(:, 22:end) - m_train) * PCALoadings(:,1:d);   
    
    MDL_1 = fitrensemble(train_inputs(:, 1:21),train_labels(:,1),'NumLearningCycles',150, 'Method', 'Bag', 'FResample',0.75);
    MDL_2 = fitrensemble(train_inputs(:, 1:21),train_labels(:,2),'NumLearningCycles',150, 'Method', 'Bag', 'FResample',0.75);
    MDL_3 = fitrensemble(train_inputs(:, 1:21),train_labels(:,3),'NumLearningCycles',150, 'Method', 'Bag', 'FResample',0.75);
    MDL_4 = fitrensemble(train_inputs(:, 1:21),train_labels(:,4),'NumLearningCycles',150, 'Method', 'Bag', 'FResample',0.75);
    MDL_5 = fitrensemble(train_inputs(:, 1:21),train_labels(:,5),'NumLearningCycles',150, 'Method', 'Bag', 'FResample',0.75);
    MDL_6 = fitrensemble(train_inputs(:, 1:21),train_labels(:,6),'NumLearningCycles',150, 'Method', 'Bag', 'FResample',0.75);
    MDL_7 = fitrensemble(train_inputs(:, 1:21),train_labels(:,7),'NumLearningCycles',150, 'Method', 'Bag', 'FResample',0.75);
    MDL_8 = fitrensemble(train_inputs(:, 1:21),train_labels(:,8),'NumLearningCycles',150, 'Method', 'Bag', 'FResample',0.75);
    MDL_9 = fitrensemble(train_inputs(:, 1:21),train_labels(:,9),'NumLearningCycles',150, 'Method', 'Bag', 'FResample',0.75);
    fprintf('built demo model \n');    
    
    % predict
    train_pred_1 = predict(MDL_1,train_inputs(:, 1:21));
    train_pred_2 = predict(MDL_2,train_inputs(:, 1:21));
    train_pred_3 = predict(MDL_3,train_inputs(:, 1:21));
    train_pred_4 = predict(MDL_4,train_inputs(:, 1:21));
    train_pred_5 = predict(MDL_5,train_inputs(:, 1:21));
    train_pred_6 = predict(MDL_6,train_inputs(:, 1:21));
    train_pred_7 = predict(MDL_7,train_inputs(:, 1:21));
    train_pred_8 = predict(MDL_8,train_inputs(:, 1:21));
    train_pred_9 = predict(MDL_9,train_inputs(:, 1:21));
    fprintf('predicted demo classes \n');
    
    train_pred_demo = [train_pred_1 train_pred_2 train_pred_3 train_pred_4 train_pred_5 train_pred_6 train_pred_7 train_pred_8 train_pred_9];
    train_error = error_metric(train_pred_demo, train_labels);
    fprintf('Demo Training error: %f\n',train_error);

    % predict
    fprintf('predicting demo test classes');
    test_pred_1 = predict(MDL_1,test_inputs(:, 1:21));
    test_pred_2 = predict(MDL_2,test_inputs(:, 1:21));
    test_pred_3 = predict(MDL_3,test_inputs(:, 1:21));
    test_pred_4 = predict(MDL_4,test_inputs(:, 1:21));
    test_pred_5 = predict(MDL_5,test_inputs(:, 1:21));
    test_pred_6 = predict(MDL_6,test_inputs(:, 1:21));
    test_pred_7 = predict(MDL_7,test_inputs(:, 1:21));
    test_pred_8 = predict(MDL_8,test_inputs(:, 1:21));
    test_pred_9 = predict(MDL_9,test_inputs(:, 1:21));

    pred_labels1 = [test_pred_1 test_pred_2 test_pred_3 test_pred_4 test_pred_5 test_pred_6 test_pred_7 test_pred_8 test_pred_9];   
    
    MDL_10 = fitrensemble(train_inputs(:, 22:end),train_labels(:,1),'NumLearningCycles',50, 'Method', 'Bag', 'FResample',0.75);
    MDL_11 = fitrensemble(train_inputs(:, 22:end),train_labels(:,2),'NumLearningCycles',50, 'Method', 'Bag', 'FResample',0.75);
    MDL_12 = fitrensemble(train_inputs(:, 22:end),train_labels(:,3),'NumLearningCycles',50, 'Method', 'Bag', 'FResample',0.75);
    MDL_13 = fitrensemble(train_inputs(:, 22:end),train_labels(:,4),'NumLearningCycles',50, 'Method', 'Bag', 'FResample',0.75);
    MDL_14 = fitrensemble(train_inputs(:, 22:end),train_labels(:,5),'NumLearningCycles',50, 'Method', 'Bag', 'FResample',0.75);
    MDL_15 = fitrensemble(train_inputs(:, 22:end),train_labels(:,6),'NumLearningCycles',50, 'Method', 'Bag', 'FResample',0.75);
    MDL_16 = fitrensemble(train_inputs(:, 22:end),train_labels(:,7),'NumLearningCycles',50, 'Method', 'Bag', 'FResample',0.75);
    MDL_17 = fitrensemble(train_inputs(:, 22:end),train_labels(:,8),'NumLearningCycles',50, 'Method', 'Bag', 'FResample',0.75);
    MDL_18 = fitrensemble(train_inputs(:, 22:end),train_labels(:,9),'NumLearningCycles',50, 'Method', 'Bag', 'FResample',0.75);
    fprintf('built model \n');    
    
    % predict
    train_pred_10 = predict(MDL_10,train_inputs(:, 22:end));
    train_pred_11 = predict(MDL_11,train_inputs(:, 22:end));
    train_pred_12 = predict(MDL_12,train_inputs(:, 22:end));
    train_pred_13 = predict(MDL_13,train_inputs(:, 22:end));
    train_pred_14 = predict(MDL_14,train_inputs(:, 22:end));
    train_pred_15 = predict(MDL_15,train_inputs(:, 22:end));
    train_pred_16= predict(MDL_16,train_inputs(:, 22:end));
    train_pred_17 = predict(MDL_17,train_inputs(:, 22:end));
    train_pred_18 = predict(MDL_18,train_inputs(:, 22:end));    
    fprintf('predicted twitter classes \n');

    train_pred_tweet = [train_pred_10 train_pred_11 train_pred_12 train_pred_13 train_pred_14 train_pred_15 train_pred_16 train_pred_17 train_pred_18];
    train_error = error_metric(train_pred_tweet, train_labels);
    fprintf('Training error: %f\n',train_error);

    % predict
    fprintf('predicting twitter test classes\n');
    test_pred_10 = predict(MDL_10,test_inputs(:, 22:end));
    test_pred_11 = predict(MDL_11,test_inputs(:, 22:end));
    test_pred_12 = predict(MDL_12,test_inputs(:, 22:end));
    test_pred_13 = predict(MDL_13,test_inputs(:, 22:end));
    test_pred_14 = predict(MDL_14,test_inputs(:, 22:end));
    test_pred_15 = predict(MDL_15,test_inputs(:, 22:end));
    test_pred_16 = predict(MDL_16,test_inputs(:, 22:end));
    test_pred_17 = predict(MDL_17,test_inputs(:, 22:end));
    test_pred_18 = predict(MDL_18,test_inputs(:, 22:end));

    pred_labels2 = [test_pred_10 test_pred_11 test_pred_12 test_pred_13 test_pred_14 test_pred_15 test_pred_16 test_pred_17 test_pred_18];   
    
    pred_labels = (pred_labels1 + pred_labels2)./2;
end
