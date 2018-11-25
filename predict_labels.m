function pred_labels = predict_labels(train_inputs,train_labels,test_inputs)

    % Function that has to be modified with your training function
    % Modify the function name below as required. An example
    % random_precitor baseline is shown below

    pred_labels = PCR(train_inputs, train_labels, test_inputs);
end
