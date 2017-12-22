function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


temp=X * theta;
temp=sigmoid(temp);
temp_theta=theta;
theta(1)=[];
J=sum(-y .* log(temp) - (1-y) .* log(1-temp))/m + (lambda/(2*m))*(sum(theta.^2));
theta=temp_theta;
temp=temp-y;
X(1:m, :)=X(1:m, :) .* temp;
grad=(sum(X, 1)/m)' + (lambda/m)*theta;
grad(1)=(sum(X, 1)/m)'(1);


% =============================================================

end
