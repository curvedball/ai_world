function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
%C = 1;
%sigma = 0.3;


% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
%C = 0.3;
%sigma = 0.1;




%param_vec = [0.01 0.03 0.1 0.3 1 3 10 30];
%param_vec = [0.3 1];
%param_vec = [0.1];
%param_vec = [0.01 0.03 0.1 0.3 1];
param_vec = [0.1 0.3 1];      %zb: 经过测试得到的最优参数是: C=1, and sigma=0.1


n = size(param_vec, 2);
p = zeros(n, n);
max_value = 0;

for i = 1 : n
    for j = 1 : n
        temp_C = param_vec(i);
        temp_sigma = param_vec(j);
        model= svmTrain(X, y, temp_C, @(x1, x2) gaussianKernel(x1, x2, temp_sigma));
        pred = svmPredict(model, Xval);
        p(i, j) = mean(double(pred == yval));   %zb: 如果是准确率，就取最大值。
        %p(i, j) = mean(double(pred ~= yval));   %zb:如果此处改为错误率，就取最小值。~=符号标号不等于, 这样p就是错误率。

        %----------------
        v = max(max(p));
        if (v > max_value)
            max_value = v;
            row_index = i;
            column_index = j;
            C = temp_C;
            sigma = temp_sigma;
        end
    end
end


%-----------------------------
p
row_index
column_index
C
sigma


% =========================================================================

end
