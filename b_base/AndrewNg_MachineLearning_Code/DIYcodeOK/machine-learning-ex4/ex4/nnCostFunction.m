function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));



% Setup some useful variables
m = size(X, 1);


% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));  %zb: set 0 with xxx rows and yyy columns
Theta2_grad = zeros(size(Theta2));



% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%





I = eye(num_labels);



%zb: 10 is the 10th neuron output! It denotes the digit '0'!!! The weight matrix is trained by this rule.
%zb: 根据列向量y的每个值在I矩阵中挑选需要的某个行，最终可以得到我们需要的矩阵。
%zb: 需要特别注意的是，手写的数字'0'在系统中是由输出层的第１０个神经元输出的，训练产生的权重矩阵Θ也是这样生成的，
%zb:    所以，y的某个值是１０，表示最后神经元的标签值应该是１
Y = I(y, :);



%zb: 按照神经网络的原理图来写
z1 = X;
a1 = [ones(m, 1) z1];
z2 = a1 * Theta1';
a2 = [ones(m, 1) sigmoid(z2)];
z3 = a2 * Theta2';
a3 = sigmoid(z3);


%zb: 没有惩罚系数lamba时的代价函数
M = Y .* log(a3) + (1 - Y) .* log(1 - a3);
J = -1 / m * sum(sum(M));



%zb: 对对权重矩阵Θ中的第一列的各个θ不要进行惩罚，因为它们是bias偏置量前面的系数。
%J = J + lambda / (2 * m) * (sum(sum(Theta1.^2)) + sum(sum(Theta2.^2)));



%zb: 有惩罚系数lamba时的代价函数
T1 = Theta1(:, 2:end);
T2 = Theta2(:, 2:end);
J = J + lambda / (2 * m) * (sum(sum(T1.^2)) + sum(sum(T2.^2)));



%zb:========================计算反向传播算法的梯度===============================
d3 = a3 - Y;
d2 = (d3 * Theta2).*(a2.*(1 - a2));


Theta2_grad = (a2'*d3)';            %zb: 本层计算出的矩阵维度刚好合适
Theta1_grad = (a1'*d2)'(2:end, :);  %zb: 简写为一行代码，反方向推进一层，需要去除计算出的矩阵的第一行

%zb: 分解成两行来写，也是可以的。
%temp = (a1'*d2)'(2:end, :);
%Theta1_grad = temp;




%zb:========================计算反向传播算法的梯度（考虑增加惩罚项）===============================
T2new = Theta2;
T2new(:, 1) = zeros(num_labels, 1); %zb: 把临时得到的权重矩阵的第一列的值设置为０，这样就可以惩罚项就可以把矩阵整体处理了。

T1new = Theta1;
T1new(:, 1) = zeros(hidden_layer_size, 1);


Theta2_grad = 1 / m * Theta2_grad + lambda / m * T2new;
Theta1_grad = 1 / m * Theta1_grad + lambda / m * T1new;



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
