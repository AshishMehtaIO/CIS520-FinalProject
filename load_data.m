function [val_input, val_labels, train_cval_input, ...
    train_cval_labels, train_cval_parts] = load_data ...
    (validation_percentage, num_folds)

    % seed random number generator
    rng(1999);

    % load the dataset
    data = importdata('training_data.mat');
    inputs = data.train_inputs;
    labels = data.train_labels;
    dataset_size = size(labels, 1);
    fprintf('Total size of data loaded: %d\n', dataset_size);

    validation_size = ceil(size(labels, 1)*validation_percentage/100);
    fprintf('Validation size: %d\n', validation_size);

    % partiton into train and validation
    rand_indices = randperm(dataset_size);
    val_indices = rand_indices(1:validation_size);
    train_indices = rand_indices(validation_size + 1: end);

    val_input = inputs(val_indices, :);
    val_labels = labels(val_indices, :);

    train_cval_input = inputs(train_indices, :);
    train_cval_labels = labels(train_indices, :);
    train_cval_size = size(train_cval_labels, 1);

    fprintf('Train+CV size: %d\n', train_cval_size);

    %shuffle the random generator
    rng('shuffle');

    % define train_cv partitions
    train_cval_parts = make_xval_partition(train_cval_size, num_folds);
    train_cval_parts = train_cval_parts';

end