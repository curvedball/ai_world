
clear ;
close all;
clc;

data = load('ex2data1.txt');
X = data(1:5, [1, 2]);
y = data(1:5, 3);
m = length(y);

X=[ones(m, 1) X];

theta = [0;0;0;]

A = sigmoid(X*theta);


X
-X
1-X


fprintf('xxxProgram paused. Press enter to continue.\n');
pause;

A
fprintf('aaaProgram paused. Press enter to continue.\n');
pause;

v=[1;2;3]
b=1/v;
b

a=[1; 1; 1];
c=1/a;
c
