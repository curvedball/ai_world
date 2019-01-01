function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
[m, n] = size(X);

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%


mu = 1 / m * (sum(X, 1))';

%zb: 运行正确的代码，但是效率不高
%for j = 1 : n
%    t = X'(j, :) - mu(j)*ones(1, m);
%    sigma2(j) = 1 / m * (t * t');
%end



t = X' - mu * ones(1, m);       %zb: 使用了一个转换的技巧, 2*1的向量，与1*m的向量相乘，刚好得到2*m的矩阵，满足我们的需求
%sigma2 = 1/ m * (t * t');      %zb: 本行有错误, 2行2列的矩阵有多余的计算在里面
sigma2 = 1 /m * sum(t.^2, 2);






% =============================================================


end
