function HLML_main(cross_validate, validate)

    % Function to load the data, splits it into training+crossvalidation and
    % validation and perform cross-validation and validation as required. 
    % Usage: HLML_main(cross_validate, validate)

    % cross_validate = [boolean] 'true' if you want to perform Cross Validation
    % validate = [boolean] 'true' if you want to perform validation
    
    %% Params

    validation_percentage = 25;
    num_folds = 10;

    %% Load data

    [val_input, val_labels, train_cval_input, train_cval_labels, ...
        train_cval_parts] = load_data(validation_percentage, num_folds);

    %% Cross validate to find good hyperparameters
    
    if cross_validate == true
        cv_error = zeros(1, max(train_cval_parts));
        for iter_num = 1:max(train_cval_parts)

            fprintf('\nCross validation iteration %d\n', iter_num);

            [Xtrain, Ytrain, XCV, YCV] = make_folds(train_cval_parts, ...
                train_cval_input, train_cval_labels, iter_num);

            YCV_hat = predict_labels(Xtrain, Ytrain, XCV);
            cv_error(iter_num) = error_metric(YCV_hat, YCV);

            fprintf("Cross validation error for iteration %d is %f", ...
            iter_num, cv_error(iter_num));
        end

        fprintf("\n\nCross validation successfully completed\n");
        fprintf("Average cross validation error: %f\n", mean(cv_error));
    end

    %% Train on the whole train_cval data and perform valiadtion on val data

    if validate == true
        val_labels_hat = predict_labels(train_cval_input, ...
        train_cval_labels, val_input);
        
        val_error = error_metric(val_labels_hat, val_labels);
        
        fprintf("\nValidation successfully completed\n");
        fprintf("Final validation error: %f\n", val_error);
    end
end