function [errorValue, delta_V, delta_W, Output] = BPOnline(Norm_Input, Norm_Output)
 
%STEP 1 : Normalize the Input
 
%Checking whether the Inputs needs to be normalized or not
 
%if max(abs(Input(:)))> 1
 
%Need to normalize
 
%Norm_Input = Input / max(abs(Input(:)));
 
%else
 
%Norm_Input = Input;
 
%end
 
%Checking Whether the Outputs needs to be normalized or not
 
%if max(abs(Output(:))) >1
 
%Need to normalize
 
%Norm_Output = Output / max(abs(Output(:)));
 
%else
 
%Norm_Output = Output;
 
%end
 
%Assigning the number of hidden neurons in hidden layer
 
m = 2;
 
%Find the size of Input and Output Vectors
 
[l,b] = size(Norm_Input);
 
[n,a] = size(Norm_Output);
 
%Initialize the weight matrices with random weights
 
V = rand(l,m); % Weight matrix from Input to Hidden
 
W = rand(m,n); % Weight matrix from Hidden to Output
 
%Setting count to zero, to know the number of iterations
 
count = 0;
 
%Calling function for training the neural network
 
[errorValue, delta_V, delta_W, Output] = trainNeuralNet(Norm_Input,Norm_Output,V,W);
 
%Checking if error value is greater than 0.1. If yes, we need to train the
 
%network again. User can decide the threshold value
 
while errorValue > 0.05
 
%incrementing count
 
count = count + 1;
 
%Store the error value into a matrix to plot the graph
 
Error_Mat(count)=errorValue;
 
%Change the weight metrix V and W by adding delta values to them
 
W=W+delta_W;
 
V=V+delta_V;
 
%Calling the function with another overload.
 
%Now we have delta values as well.
 
count;
 
[errorValue, delta_V, delta_W, Output]=trainNeuralNet(Norm_Input,Norm_Output,V,W,delta_V,delta_W);
 
end
 
%This code will be executed when the error value is less than 0.1
 
if errorValue < 0.05
 
%Incrementing count variable to know the number of iteration
 
count=count+1;
 
%Storing error value into matrix for plotting the graph
 
Error_Mat(count)=errorValue;
 
end
 
%Calculating error rate
 
Error_Rate=sum(Error_Mat)/count;
 
figure;
 
%setting y value for plotting graph
 
y=[1:count];
 
%Plotting graph
 
plot(y, Error_Mat);
 
end
 
%Function to train the network
 
%Created By : Anoop.V.S & Lekshmi B G
 
%Created On : 18-09-2013
 
%Description : Function to train the network
 
function [errorValue, delta_V, delta_W, Output] = trainNeuralNet(Input, Output, V, W, delta_V, delta_W)
 
%Function for calculation (steps 4 - 16)
 
%To train the Neural Network
 
%Calculating the Output of Input Layer
 
%No computation here.
 
%Output of Input Layer is same as the Input of Input  Layer
 
Output_of_InputLayer = Input;
 
%Calculating Input of the Hidden Layer
 
%Here we need to multiply the Output of the Input Layer with the -
 
%synaptic weight. That weight is in the matrix V.
 
Input_of_HiddenLayer = V' * Output_of_InputLayer;
 
%Calculate the size of Input to Hidden Layer
 
[m, n] = size(Input_of_HiddenLayer);
 
%Now, we have to calculate the Output of the Hidden Layer
 
%For that, we need to use Sigmoidal Function
 
Output_of_HiddenLayer = 1./(1+exp(-Input_of_HiddenLayer));
 
%Calculating Input to Output Layer
 
%Here we need to multiply the Output of the Hidden Layer with the -
 
%synaptic weight. That weight is in the matrix W
 
Input_of_OutputLayer = W'*Output_of_HiddenLayer;
 
%Clear varables
 
clear m n;
 
%Calculate the size of Input of Output Layer
 
[m, n] = size(Input_of_OutputLayer);
 
%Now, we have to calculate the Output of the Output Layer
 
%For that, we need to use Sigmoidal Function
 
Output_of_OutputLayer = 1./(1+exp(-Input_of_OutputLayer))
 
%Now we need to calculate the Error using Root Mean Square method
 
difference = Output - Output_of_OutputLayer;
 
square = difference.*difference;
 
errorValue = sqrt(sum(square(:))) %the cost function
 
%Calculate the matrix 'd' with respect to the desired output
 
%Clear the variable m and n
 
clear m n
 
[n, a] = size(Output);
 
%for i = 1 : n
 
%for j = 1 : a
%getting the delta for the last layer 
d =(Output-Output_of_OutputLayer).*Output_of_OutputLayer.*(1-Output_of_OutputLayer) %derivative of the cost function
 
%end
 
%end
 
%Now, calculate the Y matrix
 
Y = Output_of_HiddenLayer * d' %STEP 11 %This is the derivative of the cost with respect to the weights entering the hidden layer
 
%Checking number of arguments. We are using function overloading
 
%On the first iteration, we don't have delta V and delta W
 
%So we have to initialize with zero. The size of delta V and delta W will
 
%be same as that of V and W matrix respectively (nargin - no of arguments)

%to obtain the baiss
if nargin == 4
 
delta_W=zeros(size(W));
 
delta_V=zeros(size(V));
 
end
 
%Initializing eetta with 0.6 and alpha with 1
 
etta=0.6;alpha=1
 
%Calculating delta W
 
delta_W= alpha.*delta_W + etta.*Y%STEP 12
 
%STEP 13
 
%Calculating error matrix
 
error = W*d
 
%Calculating d*
 
clear m n
 
[m, n] = size(error);
 
for i = 1 : m
 
for j = 1 :n
 
d_star(i,j)= error(i,j)*Output_of_HiddenLayer(i,j)*(1-Output_of_HiddenLayer(i,j))
 
end
 
end
 
%Now find matrix, X (Input * transpose of d_star)
 
X = Input * d_star'
 
%STEP 14
 
%Calculating delta V
 
delta_V=alpha*delta_V+etta*X
 
end
