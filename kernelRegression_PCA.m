function pred_labels = kernelRegression_PCA (train_inputs, train_labels, test_inputs, k, sigma)
% Kernel regression on PCA space

% % params
d = 150;
sigma = 1.5;

pred_labels = zeros(size(test_inputs, 1), size(train_labels, 2));
X = train_inputs %- mean(train_inputs);
Xtest = test_inputs - mean(train_inputs);
Y = train_labels;
[PCALoadings,PCAScores,PCAVar] = pca(X);
X_PC = X* PCALoadings(:, 1:d);  
Xtest_PC = Xtest * PCALoadings(:, 1:d);

% X_PC = X_PC - mean(X_PC)./std(X_PC);
% Xtest_PC = Xtest_PC - mean(X_PC)./std(X_PC);

for i = 1: size(Xtest_PC, 1)
    X_t = Xtest_PC(i, :);
    X_t_rep = repmat(X_t, size(X_PC, 1), 1);
    dist = sqrt(sum((X_t_rep - X_PC).^2, 2));
    
    kernel = exp(-dist./sigma^2);
    pred_labels(i, :) = sum(Y.*kernel, 1)./sum(kernel); 
end
end