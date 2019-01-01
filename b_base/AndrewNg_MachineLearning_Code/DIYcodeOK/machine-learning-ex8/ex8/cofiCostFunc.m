function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%


% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%
%size(X)
%size(Theta)

%zb: 这是运行正确的代码
%t1 = X * Theta';
%t2 = (t1 - Y).^2;
%t3 = t2 .* R;
%J = 1 / 2 *  sum(sum(t3)); %zb: 注意，代价函数有系数1/2，但是梯度中的1/2在求导过程中消除了。



%zb: 合并为一行代码
J = 1 / 2 *  sum(sum(((X * Theta' - Y).^2) .* R));
J1 = lambda / 2 * sum(sum(Theta.^2));       %zb: 这是用户参数正则化项的代价
J2 = lambda / 2 * sum(sum(X.^2));           %zb: 这是电影样本正则化项的代价
J = J + J1 + J2;


%zb: 分步执行，依次检查矩阵的维度，然后观察我们需要的维度，来考虑如何转置
t1 = X * Theta';
t2 = t1 - Y;
t3 = t2 .* R;
X_grad = t3 * Theta;
X_grad1 = lambda * X;               %zb: 这是正则化项
X_grad = X_grad + X_grad1;


%zb: 可以使用前面已经计算出来的矩阵，提高程序执行效率
t3 = t2 .* R;
Theta_grad = t3' * X;
Theta_grad1 = lambda * Theta;       %zb: 这是正则化项
Theta_grad = Theta_grad + Theta_grad1;







% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
