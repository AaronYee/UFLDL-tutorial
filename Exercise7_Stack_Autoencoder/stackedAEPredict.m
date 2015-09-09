function [pred] = stackedAEPredict(theta, inputSize, hiddenSize, numClasses, netconfig, data)
                                         
% stackedAEPredict: Takes a trained theta and a test data set,
% and returns the predicted labels for each example.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 

% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
%% Unroll theta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.

a1 = data;                     % inputSize * M
% 
z2 = stack{1}.w * a1 + repmat(stack{1}.b,1,size(a1,2));  % hiddenSizeL1 * M
a2 = sigmoid(z2);
% 
z3 = stack{2}.w * a2 + repmat(stack{2}.b,1,size(a2,2));      % hideenSizeL2 * M
a3 = sigmoid(z3);

z4 = softmaxTheta * a3; 
preData = bsxfun(@minus, z4, max(z4, [], 1));         % numClasses * M                                           
hypothesis = bsxfun(@rdivide, exp(preData), sum(exp(preData)));    

[value, pred] = max(hypothesis);



% -----------------------------------------------------------

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
