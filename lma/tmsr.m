function [result, bestNet, worstNet] = tmsr(dataset, nInputs, xLabel, yLabel, pltTitle, name)

    nHidden = {5 4 7 10 12 15 20};
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

            % Setup division of data for training, validation, testing
            net.divideFcn = 'divideblock'; 
            net.divideParam.trainRatio = 70/100;
            net.divideParam.valRatio = 15/100;
            net.divideParam.testRatio = 15/100;

            % Training parameters
            net.trainParam.goal = 0; 
            net.trainParam.mu=1.0000e-003;
            net.trainParam.mu_inc=10;
            net.trainParam.mu_dec=1; 
            net.trainParam.epochs =5000;		
            net.trainParam.max_fail=5000;

            % Choose a performance function
            net.performFcn = 'mse';  % Mean Squared Error

            % Setup initial weights
            net = configure(net,x,t);
            net.IW{1,1} = initWeigths(nInputs, hiddenLayerSize, nInputs*2);
            net.IW{1,2} = initWeigths(nInputs, hiddenLayerSize, 2);
            net.LW{2,1} = initWeigths(nInputs, 1, hiddenLayerSize);
            
            % Export data
            save(strcat('weights/',name,'/',num2str(hiddenLayerSize),'/weights_init_',num2str(j),'.mat'),'net')
            
            % Train the network
            [net, tr] = train(net,x,t,xi,ai); 
            NET{i,j} = net;
            
            % Export data
            save(strcat('weights/',name,'/',num2str(hiddenLayerSize),'/weights_final_',num2str(j),'.mat'),'net')

            % Test the network and compute metrics
            y = net(x,xi,ai);
            testTargets = gmultiply(t,tr.testMask);
            MSE(i,j) = perform(net,testTargets,y);
            for k = 1:length(testTargets)
                if ~isnan(cell2mat(testTargets(k)))
                    break
                end
            end
            Cyt = corrcoef(cell2mat(y(k:end)),cell2mat(testTargets(k:end)));
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