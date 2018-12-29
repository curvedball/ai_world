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



%=========================================================================================
temp_value = 0;
%zb: s个样本，ｉ表示神经元，ｊ表示上一层的神经元，层数自己手工写
for s = 1 : m
    temp_a1 = a1(s, :);
    temp_a2 = a2(s, :);
    temp_a3 = a3(s, :);
    temp_Y = Y(s, :);
    temp_delta3 = temp_a3 - temp_Y;     %zb: temp_delta3是一个样本的行向量

    %可能有错误的一行
    %temp_delta2 = (temp_delta3 * Theta2).*sigmoid(temp_a2); %zb: temp_delta2也是一个样本的反向传播后的行向量

    %仍然不对
    %temp_delta2 = (temp_delta3 * Theta2).*sigmoidGradient(temp_a2);
    %temp_delta2 = temp_delta2(1, 2:end);


    %
    %temp_v = (temp_delta3 * Theta2);
    %temp_v = temp_v(1, 1:end);
    %temp_delta2 = temp_v.*sigmoidGradient(temp_a2(1, 1:end));
    %temp_delta2 = temp_delta2(2:end);


    %temp_delta2 = (temp_delta3 * Theta2) .*(temp_a2.*(1-temp_a2));
    %temp_delta2(1, 1) = 0;

    temp_u = (temp_delta3 * Theta2);
    temp_v = (temp_a2.*(1-temp_a2));
    temp_delta2 = temp_u .* temp_v;
    temp_delta2 =temp_delta2(2:end);



    %size(temp_delta3)
    %size(temp_a2)

    %处理第２层到第3层
    for i = 1 : num_labels
        for j = 1 : hidden_layer_size + 1
            temp_value = temp_a2(1, j) * temp_delta3(1, i);
            Theta2_grad(i, j) = Theta2_grad(i, j) + temp_value;
        end
    end


     %处理第1层到第2层
    for i = 1 : hidden_layer_size
        for j = 1 : input_layer_size + 1
            temp_value = temp_a1(1, j) * temp_delta2(1, i);
            Theta1_grad(i, j) = Theta1_grad(i, j) + temp_value;
        end
    end
end


Theta2_grad = 1 / m * Theta2_grad;
Theta1_grad = 1 / m * Theta1_grad;



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end