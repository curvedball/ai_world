function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


%fprintf('m=%d\n n=%d', m, size(X, 2));


%hidden_neurons = size(Theta1, 1);


%================zb: 97.52%===============
X = [ones(m, 1) X];
a2 = [ones(m, 1) sigmoid(X*Theta1')];
a3 = sigmoid(a2 * Theta2');
[value, p] = max(a3, [], 2);        %zb: return the [value, index] of the max element of the current row!



%================zb: 69.62%===============
%X = [ones(m, 1) X];
%a2 = [ones(m, 1) X*Theta1'];
%a3 = a2 * Theta2';
%[value, p] = max(a3, [], 2);


%================zb: 97.52%, So it is OK without sigmoid for the output values!===============
%X = [ones(m, 1) X];
%a2 = [ones(m, 1) sigmoid(X*Theta1')];
%a3 = a2 * Theta2';
%[value, p] = max(a3, [], 2);




% =========================================================================


end
