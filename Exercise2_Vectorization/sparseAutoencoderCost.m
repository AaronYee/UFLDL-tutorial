function [cost, grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                              lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1 : visibleSize * hiddenSize), hiddenSize, visibleSize);
W2 = reshape(theta((visibleSize * hiddenSize + 1) : 2 * visibleSize * hiddenSize), visibleSize, hiddenSize);
b1 = theta((2 * visibleSize * hiddenSize + 1) : (2 * visibleSize * hiddenSize + hiddenSize));
b2 = theta((2 * visibleSize * hiddenSize + hiddenSize + 1) : end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 

cost = 0;
W1grad = zeros(size(W1));
W2grad = zeros(size(W2));
b1grad = zeros(size(b1));
b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 

m = size(data,2);
a1 = data;
z2 = bsxfun(@plus, W1 * a1, b1);
a2 = sigmoid(z2);
z3 = bsxfun(@plus, W2 * a2, b2);
a3 = sigmoid(z3);

p = sparsityParam;
pMean = mean(a2,2);

errorTerm = mean(sum((a1 - a3) .^ 2)) / 2;
decayTerm = sum(sum(W1 .^ 2)) + sum(sum(W2 .^ 2));
sparseTerm = sum(p .* log(p ./ pMean) + (1 - p) .* log((1 - p) ./ (1 - pMean)));

cost = errorTerm + lambda / 2 * decayTerm + beta * sparseTerm;


delta3 = -(a1 - a3) .* a3 .* (1 - a3);
sparseGrad = (1 - p) ./ (1 - pMean) - p ./ pMean;
delta2 = (bsxfun(@plus, W2' * delta3, beta * sparseGrad)) .* a2 .* (1 - a2);

W1grad = W1grad + delta2 * a1';
W2grad = W2grad + delta3 * a2';

b1grad = b1grad + mean(delta2,2);
b2grad = b2grad + mean(delta3,2);

W1grad = W1grad / m + lambda * W1;
W2grad = W2grad / m + lambda * W2;

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end

