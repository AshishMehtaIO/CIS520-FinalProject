function pred_labels = autoEncoder (train_inputs, train_labels, test_inputs)
    % Autoencoder
    hiddenSize = 200;
    X_train = train_inputs';
    X_test = test_inputs';
    autoenc = trainAutoencoder(X_train, 'hiddenSize',hiddenSize,... 
    'L2WeightRegularization', 1e-3,'MaxEpochs', 200, ...
    'SparsityProportion', 0.01, 'ScaleData', true);  

    save autoenc

    train_encoded = encode(autoenc, X_train); 
    test_encoded = encode(autoenc, X_test);
    % p x n
    train_encoded = train_encoded';
    test_encoded = test_encoded';
    
    train_inputs_ones = [ones(size(train_encoded, 1), 1), train_encoded];
    test_inputs_ones = [ones(size(test_encoded, 1), 1), test_encoded];
    weights = zeros(size(train_inputs_ones, 2), size(train_labels, 2));
    for i = 1:size(train_labels, 2)
        weights(:, i) = regress(train_labels(:, i), train_inputs_ones);
    end
    
    train_pred = train_inputs_ones * weights;
    train_error = error_metric(train_pred, train_labels);
    fprintf('Training error: %f\n',train_error);
    
    pred_labels = test_inputs_ones * weights;
end