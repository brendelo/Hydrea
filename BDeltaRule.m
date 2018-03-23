function [W, startError, endError, allErrors] = Train(Input, W, D, n)
 A = Input * W; %obtaining the activation function
 O = 1 ./ (1 + exp(-A));% obtaining the first output prior to iteration
startError = norm(O - D);
allErrors = []; %added this for the error plot
for epoch=1:n
    A = Input * W;
    O = 1 ./ (1 + exp(-A));
    %dW = mean(2*(O-D).*O.*(1-O).*Input)';
    dW = Input'*(2*(O-D).*O.*(1-O)); % this is for a GENERAL single layer neural network
    W = W - 0.1*dW;
    endError = norm(1 ./ (1 + exp(-(Input * W))) - D);
    allErrors = [allErrors;endError]; % added this for the error plot
end
