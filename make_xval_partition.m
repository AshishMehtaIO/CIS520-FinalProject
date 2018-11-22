function [part] = make_xval_partition(n, n_folds)

% MAKE_XVAL_PARTITION - Randomly generate cross validation partition.
%
% Usage:
%
%  PART = MAKE_XVAL_PARTITION(N, N_FOLDS)
%
% Randomly generates a partitioning for N datapoints into N_FOLDS equally
% sized folds (or as close to equal as possible). PART is a 1 X N vector,
% where PART(i) is a number in (1...N_FOLDS) indicating the fold assignment
% of the i'th data point.

% YOUR CODE GOES HERE

% PROCEDURE:
%
% (1). Calculate the number of repeats(called numCoverData) there would be,
% if the fold size was exactly 1.
% (2). Assign fold id (ranging from 1:n_fold) sequentially to all examples in 
% a round-robin fashion. This means that if n_folds is 2, assign id 1 to x_1,
% 2 to x_2 , 1 to x_3 , 2 to x_4 and so on until x_n. That is, repeat
% 1:n_fold in a verctor numCoverData times.
% (3). Randomize these id assignments! (i.e. Shuffle)

% compute the number of times we need to train-test on a given n_fold 
% selection to cover all parts/folds as both train and test parts/folds. 
numCoverData = ceil(n/n_folds);

% generate a vector that holds numCoverData times the n fold-assignments
% on each observation in the Dataset
part = repmat(1:n_folds, 1, numCoverData); 

% Truncate to size = number of Data samples-n / All fold_sets may not have
% equal # of parts since mod(n,n_folds) may not be 0.
part = part(1:n);

% Make this partitoin of datapoints random i.e. randomly assign them fold
% ids
part = part(randperm(n));