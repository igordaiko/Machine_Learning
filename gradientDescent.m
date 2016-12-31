function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
theta_history = zeros(num_iters, 2);
for iter = 1:num_iters
  
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    % ============================================================
    a = 0;
    buf = zeros(length(theta), 1);
    for j = 1:2
      for i = 1:3
        a = a + (theta(1) + theta(2)*X(i,2) - y(i))*X(i,j);
      end
      buf(j) = theta(j) - (alpha*a)/m;    
    end
    theta = buf;
    %a = sum((theta'.*X - y).*X); 
    %a = alpha*a/m;
    %theta = (theta - a)(1,:)';
    
    theta_history(iter, :) = theta';
    
    
    J_history(iter) = computeCost(X, y, theta);
    
    if(iter >=2)
      if(J_history(iter) > J_history(iter-1))
        break;
      end
    end
end

end
