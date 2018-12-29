function W = debugInitializeWeights(fan_out, fan_in)
%DEBUGINITIALIZEWEIGHTS Initialize the weights of a layer with fan_in
%incoming connections and fan_out outgoing connections using a fixed
%strategy, this will help you later in debugging
%   W = DEBUGINITIALIZEWEIGHTS(fan_in, fan_out) initializes the weights 
%   of a layer with fan_in incoming connections and fan_out outgoing 
%   connections using a fix set of values
%
%   Note that W should be set to a matrix of size(1 + fan_in, fan_out) as
%   the first row of W handles the "bias" terms
%



% Set W to zeros
%zb: 根据对神经网络原理图的分析，
%zb: 扇出（下一层的神经元的个数）对应这权重矩阵Θ的行数，
%zb: 扇入(上一层的神经元的个数)和１个偏置量的数目对应权重矩阵Θ的列数
W = zeros(fan_out, 1 + fan_in);


% Initialize W using "sin", this ensures that W is always of the same
% values and will be useful for debugging

%zb: sin(1:10) can generate 10 values! size(W) e.g. returns (5, 8)
%zb: reshape函数需要３个参数，第一个参数是一个长的行向量，第二个和第三个参数表示需要组件的矩阵的行数和列数
%zb: size(W)返回的一个二元组，刚好是行数和列数
W = reshape(sin(1:numel(W)), size(W)) / 10;

% =========================================================================

end
