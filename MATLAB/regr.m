function [result, bestNet, worstNet] = regr(dataset, nInputs, name)

    nHidden = {4 7 10 12 15 20};
    nRep = 10;
    
    % Process the data
    t = dataset(:,end)';
    x = dataset(:,1:nInputs)';
  
    % Set the optimization algorithm
    trainFcn = 'trainlm'; % Levenberg-Marquardt optimization algorithm

    % Initialise arrays for performance
    NET = cell(length(nHidden), nRep);
    MSE = zeros(length(nHidden), nRep);
    R = zeros(length(nHidden), nRep);
    
    for i = 1:length(nHidden)      

        for j = 1:nRep

            % Set activation functions
            hiddenLayerSize = nHidden{i};

            % Create the ANN
            net = fitnet(hiddenLayerSize,trainFcn);

            % Set activation functions
            net.inputs{1}.processFcns = {'mapstd'}; 
            net.outputs{2}.processFcns = {'mapstd'};

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
            net.IW{1,1} = initWeigths(nInputs, hiddenLayerSize, nInputs);
            net.LW{2,1} = initWeigths(nInputs, 1, hiddenLayerSize);
            
            % Export data
            iw = net.IW;
            lw = net.LW;
            b = net.b;
            save(strcat('weights/',name,'/',num2str(hiddenLayerSize),'/weights_init_',num2str(j),'.mat'),'iw', 'lw', 'b')
            
            % Train the network
            [net,~] = train(net,x,t); 
            NET{i,j} = net;
            
            % Export data
            iw = net.IW;
            lw = net.LW;
            b = net.b;
            save(strcat('weights/',name,'/',num2str(hiddenLayerSize),'/weights_final_',num2str(j),'.mat'),'iw', 'lw', 'b')

            % Test the network and compute metrics
            y = net(x);
            e = gsubtract(t,y);
            MSE(i,j) = mse(e);
            ssres = sum(e.^2);
            ybar = mean(t);
            sstot = sum(y-ybar).^2;
            rsq = 1 - (ssres/sstot);            
            R(i,j) = sqrt(rsq);
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
    iw = bestNet.IW;
    lw = bestNet.LW;
    b = bestNet.b;
    save(strcat('weights/',name,'/weights_best.mat'),'iw', 'lw', 'b')

    iw = worstNet.IW;
    lw = worstNet.LW;
    b = worstNet.b;
    save(strcat('weights/',name,'/weights_worst.mat'),'iw', 'lw', 'b')
    
   return 

end