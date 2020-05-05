t = target';
x = input';
ninputs = 11;


% Set activation functions
hiddenLayerSize = 4;

% Create the ANN
net = fitnet(hiddenLayerSize);

% Choose a performance function
net.performFcn = 'mse';  % Mean Squared Error

% Scale the data: means to 0 and deviations to 1
net.inputs{1}.processFcns = {'mapstd'}; 
net.outputs{2}.processFcns = {'mapstd'};

% Configure the network
net = configure(net,x,t);


% Divide the data for training, validation, testing
trainRatio = 70/100;
valRatio = 15/100;
testRatio = 15/100;
[trainInd,valInd,testInd] = divideblock(length(dataset),trainRatio,valRatio,testRatio);
train = dataset(trainInd,:);
trainX = train(:,1:end-1);
trainT = train(:,end);
validation = dataset(valInd,:);
validationX = validation(:,1:end-1);
validationT = validation(:,end);
test = dataset(testInd,:);
testX = test(:,1:end-1);
testT = test(:,end);


pso(net, hiddenLayerSize, ninputs, trainX, trainT, testX, testT)

