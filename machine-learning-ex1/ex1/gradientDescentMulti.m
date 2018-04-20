function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    #{
    h = X*theta;

    theta_length = length(theta);
    
    for i  = 1: theta_length,
        theta_original(i) = theta(i) - alpha * (1/m) * sum((h - y) .* X(:, i));
    end
    
    theta = theta_original;
    #}
    
    % reference for vectorize implementation of multi variable gradient descent
    %https://stackoverflow.com/questions/20736460/vectorizing-a-gradient-descent-algorithm
    gradient = (alpha /m) * X' * (X*theta - y);
    
    theta = theta - gradient;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
