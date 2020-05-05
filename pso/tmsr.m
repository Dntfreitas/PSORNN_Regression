function [result, bestNet, worstNet] = tmsr(dataset, nInputs, xLabel, yLabel, pltTitle, name)

    nHidden = {4 7 10 12 15 20};
    nRep = 10;
      
    % Process the data
    T = dataset(:,end)';
    inputs = dataset(:,1:nInputs);
    [m,~] = size(inputs);
    X = cell(1,m);
    for i = 1:m
        X{1,i} = cell2mat({inputs{i,1:end}})';
    end
     
    % Set the optimization algorithm
    trainFcn = 'trainlm'; % Levenberg-Marquardt optimization algorithm

    % Create a nonlinear autoregressive network with external input
    inputDelays = 1:2;
    feedbackDelays = 1:2;
     
    % Initialise arrays for performance
    NET = cell(length(nHidden), nRep);
    MSE = zeros(length(nHidden), nRep);
    R = zeros(length(nHidden), nRep);
    
    for i = 1:length(nHidden)      

        for j = 1:nRep

            % Set activation functions
            hiddenLayerSize = nHidden{i};

            % Create the ANN
            net = narxnet(inputDelays,feedbackDelays,hiddenLayerSize,'open',trainFcn);

            % Set activation functions
            net.inputs{1}.processFcns = {'mapstd'}; 
            net.outputs{2}.processFcns = {'mapstd'};

            % Prepare the data for training and simulation
            [x,xi,ai,t] = preparets(net,X,{},T);
            
            % Configure the network
            net = configure(net,x,t);

           % Divide the data for training, validation, testing
            trainRatio = 70/100;
            valRatio = 15/100;
            testRatio = 15/100; 
            [trainInd,valInd,testInd] = divideblock(length(t),trainRatio,valRatio,testRatio);
            trainX = x(:,trainInd)';
            trainT = t(:,trainInd)';
            validationX = x(:,valInd)';
            validationT = t(:,valInd)';
            testX = x(:,testInd)';
            testT = t(:,testInd)';
           
            % Choose a performance function
            net.performFcn = 'mse';  % Mean Squared Error

            % Configure the network
            net = configure(net,x,t);
            
            % Train the network
            net = psotmsr(net, hiddenLayerSize, nInputs, trainX, trainT, validationX, validationT, xi,ai);
            NET{i,j} = net;
            
            % Export data
            save(strcat('weights/',name,'/',num2str(hiddenLayerSize),'/weights_final_',num2str(j),'.mat'),'net')

            % Test the network and compute metrics
            y = net(testX',xi,ai);
            e = gsubtract(testT',y);
            MSE(i,j) = mse(e);
            testT = cell2mat(testT);
            y = cell2mat(y)';
            Cyt = corrcoef(testT,y');
            R(i,j) = Cyt(2,1);
            clear net
        end
    end

    % Compute table with results
    result = results(nHidden, MSE, R);
    resultCells = num2cell(result);
    header = {'No. hidden','Best MSE', 'R', 'Worst MSE', 'R', 'Mean MSE', 'STD'};
    outputXLS = [header; resultCells];
    xlswrite(strcat(name,'.xls'), outputXLS);

    % Get best and worst MSE
    bestMse = min(min(MSE));
    worstMse = max(max(MSE));

    % Get the best ANN
    [Pos_row,Pos_colum]=find(MSE == bestMse);   
    bestNet = NET{Pos_row,Pos_colum};
    
    % Get the worst ANN
    [Pos_row,Pos_colum]=find(MSE == worstMse);   
    worstNet = NET{Pos_row,Pos_colum};
  
    % Export data
    save(strcat('weights/',name,'/weights_best.mat'),'bestNet')
    save(strcat('weights/',name,'/weights_worst.mat'),'worstNet')

    % Test the Network
    y = bestNet(x,xi,ai);

    % Closed loop network
    netc = closeloop(bestNet);
    [xc,xic,aic,~] = preparets(netc,X,{},T);
    yc = netc(xc,xic,aic);

    % Step-Ahead prediction
    nets = removedelay(bestNet);
    [xs,xis,ais,~] = preparets(nets,X,{},T);
    ys = nets(xs,xis,ais);

    % Arrange time series
    ts0 = timeseries(cell2mat(T),1:length(X));
    ts1 = timeseries(cell2mat(y),3:length(X));
    ts2 = timeseries(cell2mat(yc),3:length(X));
    ts3 = timeseries(cell2mat(ys),3:length(X)+1);

    % Draw the plots
    figure
    plot(ts0)
    hold on 
    plot(ts1)
    plot(ts2)
    plot(ts3)
    title(pltTitle)
    xlabel(xLabel)
    ylabel(yLabel)
    legend({'Real','Open Loop','Closed Loop','Step-Ahead'}, 'Location','northwest')
    grid on
    hold off

    % Save the plots
    savefig(strcat('img/',name,'.fig'));
    
   return 

end