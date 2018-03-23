
 
%Created By : Odigwe Brendan Elochukwu
 
%Created On : 09-03-2018

function [errorValue,EndErrorValue,W, V, Output_of_Output, Hidden_Output] = BackPropagation(Input,Output,epo,hn) 
 %hn = number ofhidden layers
 %epo = number of iterations (epochs)
 
%Find the size of Input and Output Vectors, so that the weights can be
%modelled accordingly
if max(abs(Input(:)))> 1
    Input = Input / max(abs(Input(:)));
end
[l,b] = size(Input);
%Output = DXor;
[n,a] = size(Output);
 
%Initialize the weight matrices with random weights
 
V = rand(b,hn) % Weight matrix from Input to Hidden
 
W = rand(hn+1,a) % Weight matrix from Hidden to Output


%Calling function for training the neural network
 
[errorValue,EndErrorValue,W, V, Output_of_Output, Hidden_Output,Errors,count] = trainNeuralNet(Input,Output,V,W,epo,hn);
y = [1:count];
hold off
plot(y,Errors);

title('Plot to show the decrease of the Error value with every Iteration');
ylabel('Error values');
xlabel('Epoch(s)');
 
function [errorValue, EndErrorValue, W, V, Output_of_Output,Hidden_Output,Errors,count] = trainNeuralNet(Input, Output, V, W,epo,hn)
sizeofV = size(V)
sizeofW = size(W)

[l,b] = size(Input);
 
[n,a] = size(Output);  
Output_of_Input = Input;
Lrate = 0.5; 
%Calculating Input of the Hidden Layer
Hidden_Input = Output_of_Input * V;
Hidden_Output = 1./(1+exp(-Hidden_Input));
Hidden_Output = [Hidden_Output ones(l,1)]; %adding a column of ones to the output of hiddenlayer 


Input_of_Output = Hidden_Output * W;
Output_of_Output = sigmoid(Input_of_Output); 
errorValue = norm(Output - Output_of_Output); %the cost function
count = 0;
for epoch=1:epo
    count = count+1;
     
    dEdW = Hidden_Output' *((Output_of_Output-Output).*Output_of_Output.*(1-Output_of_Output)); 
    %derivative of the cost function with repect to the weights entering
    %the output layer fromthe hidden layer
    
    [e,f] = size(W);
    [g,h] = size(Hidden_Output);
    e= e-1;
    h= h-1;
    Hidden_Output = Hidden_Output(:,[1:h]);
    W2 = W([1:e],:); %removing the row for multiplication with the threshold/bias initially put in 
    dEdV = Input'*(((Output_of_Output-Output).*Output_of_Output.*(1-Output_of_Output)*W2').*(Hidden_Output.*(1-Hidden_Output)));
    %This is the derivative of the cost function with repect to the weights entering the hidden layer
    V = V - Lrate * dEdV;
    W = W - Lrate * dEdW;
    Hidden_Output = 1./(1+exp(-(Output_of_Input * V)));
    Hidden_Output = [Hidden_Output ones(l,1)]; %adding a column of ones to the output of hiddenlayer 

    Output_of_Output = 1./(1+exp(-(Hidden_Output * W))); %sigmoid
    EndErrorValue = norm(Output - Output_of_Output); %the cost function  
    Errors(count) = EndErrorValue;
end



