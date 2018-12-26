function plotDecisionBoundary(theta, X, y)
%PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
%the decision boundary defined by theta
%   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the 
%   positive examples and o for the negative examples. X is assumed to be 
%   a either 
%   1) Mx3 matrix, where the first column is an all-ones column for the 
%      intercept.
%   2) MxN, N>3 matrix, where the first column is all-ones

% Plot Data
plotData(X(:,2:3), y);


hold on


if size(X, 2) <= 3 %zb: plot a line for the ex2data1.txt
    % Only need 2 points to define a line, so choose two endpoints
    plot_x = [min(X(:,2))-2,  max(X(:,2))+2]; %zb: Set the x values of two points

    % Calculate the decision boundary line
    %zb: theta0+theta1*x1 + theta2*x2=0 is the boundary. So we can calculate the x2 values
    %zb: consider x1 as the x value, and x2 as the y value.
    % x2 = (theta0 + theta1*x1) / (-theta2)
    %
    %plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1)); %zb: OK
    %plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1)); %zb: also OK, the scalar value of theta(1) is converted!
    plot_y = (-1./theta(3)).*(theta(2)*plot_x + theta(1));   %zb: still OK, * is also OK to substitue .*
    plot_y = (-1/theta(3)).*(theta(2)*plot_x + theta(1));    %zb: OK, / to substitue ./


    % Plot, and adjust axes for better viewing
    plot(plot_x, plot_y, 'g-', 'linewidth', 5)

    % Legend, specific for the exercise
    legend('Admitted', 'Not admitted', 'Decision Boundary')
    axis([30, 100, 30, 100])
else
    % Here is the grid range
    u = linspace(-1, 1.5, 50);
    v = linspace(-1, 1.5, 50);

    z = zeros(length(u), length(v));

    % Evaluate z = theta*x over the grid
    for i = 1:length(u)
        for j = 1:length(v)
            z(i,j) = mapFeature(u(i), v(j))*theta;
        end
    end
    z = z'; % important to transpose z before calling contour

    % Plot z = 0
    % Notice you need to specify the range [0, 0]
    %contour(u, v, z, [0, 0], 'LineWidth', 2)   %zb: tell contour to connect some points which have the same value!
    %contour(u, v, z, [0, 0.5, 1], 'LineWidth', 2)
    %contour(u, v, z, [0, 0.1, 1], 'LineWidth', 2)
    %contour(u, v, z, [0, 0.2, 1], 'LineWidth', 2)
    contour(u, v, z, [0, 1], 'LineWidth', 2)

    %contour(u, v, z, [2, 2], 'LineWidth', 2) %There are no values of z==2, So we cannot see the curve!

end


hold off

end
