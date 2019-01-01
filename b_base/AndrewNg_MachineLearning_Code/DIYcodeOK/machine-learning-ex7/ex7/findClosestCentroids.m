function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1); %zb:  注意，这里设置的idx是一个列向量

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

m = size(X,1);
for i = 1 : m
    for j = 1 : K
        v = X(i, :) - centroids(j, :);  %zb: 获取矩阵的某行必须设置列不关心，冒号是绝对不能省略的！
        d(j) = v * v';
    end
    [val, idx(i)] = min(d);     %zb: 求行向量或者列向量的最小值，不需要指明行或者列，返回的就是合适的元素值与索引值
end



% =============================================================

end

