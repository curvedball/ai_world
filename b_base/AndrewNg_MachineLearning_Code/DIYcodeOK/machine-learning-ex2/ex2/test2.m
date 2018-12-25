
clear;
close all;
clc;


u = linspace(-1, 1.5, 50);
v = linspace(-1, 1.5, 50);


%fprintf('Pause. Press any key to continue!\n');
%pause;

z = zeros(length(u), length(v));

m=size(z, 1);
n=size(z, 2);
%fprintf('m=%d\n', m);
%fprintf('n=%d\n', n);



% Evaluate z = theta*x over the grid
for i = 1:length(u)
    for j = 1:length(v)
        z(i,j) = mapFeature(u(i), v(j))*theta;
    end
end
z = z'; % important to transpose z before calling contour

% Plot z = 0
% Notice you need to specify the range [0, 0]
contour(u, v, z, [0, 0], 'LineWidth', 2)
%contour(u, v, z, [1, 2], 'LineWidth', 2)
