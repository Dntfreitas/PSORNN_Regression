nInputs = 2;
xLabel = 'Month';
yLabel = 'Pounds per cow';
pltTitle = 'Monthly Milk Production';
name = 'milk';
[r1, bestNet1, worstNet1] = tmsr(milk, nInputs, xLabel, yLabel, pltTitle, name);

nInputs = 17;
xLabel = 'Hour of day';
yLabel = 'Slowness in traffic (%)';
pltTitle = 'Urban Traffic in São Paulo - Brazil';
name = 'traffic';
[r2, bestNet2, worstNet2] = tmsr(traffic, nInputs, xLabel, yLabel, pltTitle, name);

function [result, bestNet, worstNet] = tmsr(dataset, nInputs, xLabel, yLabel, pltTitle, name)

    nHidden = {4 7 10 12 15 20};
    nRep = 2;
    
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
            net.trainParam.epochs =100;		
            net.trainParam.max_fail=100;

            % Choose a performance function
            net.performFcn = 'mse';  % Mean Squared Error

            % Setup initial weights
            net = configure(net,x,t);
            net.IW{1,1} = initWeigths(nInputs, hiddenLayerSize, nInputs*2);
            net.IW{1,2} = initWeigths(nInputs, hiddenLayerSize, 2);
            net.LW{2,1} = initWeigths(nInputs, 1, hiddenLayerSize);
            
            % Export data
            iw = net.IW;
            lw = net.LW;
            b = net.b;
            save(strcat('weights/',name,'/',num2str(hiddenLayerSize),'/weights_init_',num2str(j),'.mat'),'iw', 'lw', 'b')
            
            % Train the network
            [net,~] = train(net,x,t,xi,ai); 
            NET{i,j} = net;
            
            % Export data
            iw = net.IW;
            lw = net.LW;
            b = net.b;
            save(strcat('weights/',name,'/',num2str(hiddenLayerSize),'/weights_final_',num2str(j),'.mat'),'iw', 'lw', 'b')

            % Test the network and compute metrics
            y = net(x,xi,ai);
            e = gsubtract(t,y);
            e = cell2mat(e);
            MSE(i,j) = mse(e);
            ssres = sum(e.^2);
            ybar = mean(cell2mat(t));
            sstot = sum((cell2mat(y)-ybar).^2);
            rsq = 1 - (ssres/sstot);            
            R(i,j) = sqrt(rsq);
            clear net tr
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
    iw = bestNet.IW;
    lw = bestNet.LW;
    b = bestNet.b;
    save(strcat('weights/',name,'/weights_best.mat'),'iw', 'lw', 'b')

    iw = worstNet.IW;
    lw = worstNet.LW;
    b = worstNet.b;
    save(strcat('weights/',name,'/weights_worst.mat'),'iw', 'lw', 'b')

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

function [result] = results(nHidden, MSE, R)

    % Initialise the result array
    result = zeros(length(nHidden), 7);
   
    for i = 1:length(nHidden)
        mseAux = MSE(i,:);
        rAux = R(i,:);
        
        % Get best and worst MSE
        bestMse = min(min(mseAux));
        worstMse = max(max(mseAux));
        
        % Get the best ANN
        [~,PosBest] = find(mseAux == bestMse);   
     
        % Get the worst ANN
        [~,PosWorst]=find(mseAux == worstMse);   
      
        % Compose result
        result(i,1) = nHidden{i};
        result(i,2) = bestMse;
        result(i,3) = rAux(PosBest);
        result(i,4) = worstMse;
        result(i,5) = rAux(PosWorst);
        result(i,6) = mean(mseAux);
        result(i,7) = std(mseAux);
        
    end   

end

function [w] = initWeigths(nInputs, sz1, sz2)
    l = 2.4 * nInputs;
    a = -l;
    b = l;
    w = (b-a).*rand(sz1, sz2) + a;
end