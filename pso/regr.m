function [result, bestNet, worstNet] = regr(dataset, nInputs, name)

    nHidden = {4 7 10 12 15 20};
    nRep = 10;
        
    % Process the data
    t = dataset(:,end)';
    x = dataset(:,1:nInputs)';

    % Initialise arrays for performance
    NET = cell(length(nHidden), nRep);
    MSE = zeros(length(nHidden), nRep);
    R = zeros(length(nHidden), nRep);
    
    for i = 1:length(nHidden)      

        for j = 1:nRep

            % Set activation functions
            hiddenLayerSize = nHidden{i};

            % Create the ANN
            net = fitnet(hiddenLayerSize);

            % Set activation functions
            net.inputs{1}.processFcns = {'mapstd'}; 
            net.outputs{2}.processFcns = {'mapstd'};

            % Configure the network
            net = configure(net,x,t);
            
            % Divide the data for training, validation, testing
            trainRatio = 70/100;
            valRatio = 15/100;
            testRatio = 15/100;
            [trainInd,valInd,testInd] = divideblock(length(t),trainRatio,valRatio,testRatio);
            train = dataset(trainInd,:);
            trainX = train(:,1:end-1)';
            trainT = train(:,end)';
            validation = dataset(valInd,:);
            validationX = validation(:,1:end-1)';
            validationT = validation(:,end)';
            test = dataset(testInd,:);
            testX = test(:,1:end-1)';
            testT = test(:,end)';


            % Choose a performance function
            net.performFcn = 'mse';  % Mean Squared Error
            
            % Train the network
            net = psoregr(net, hiddenLayerSize, nInputs, trainX, trainT, validationX, validationT); 
            NET{i,j} = net;
            
            % Export data
            save(strcat('weights/',name,'/',num2str(hiddenLayerSize),'/weights_final_',num2str(j),'.mat'),'net')

            % Test the network and compute metrics
            y = net(testX);
            e = gsubtract(testT,y);
            MSE(i,j) = mse(e);
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
    
   return 

end