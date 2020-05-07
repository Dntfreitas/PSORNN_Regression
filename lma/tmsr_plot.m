function tmsr_plot(dataset, nInputs, net_org, hiddenLayerSize)

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
     
   
    % Create the ANN
    net = narxnet(inputDelays,feedbackDelays,hiddenLayerSize,'open',trainFcn);

    % Set activation functions
    net.inputs{1}.processFcns = {'mapstd'}; 
    net.outputs{2}.processFcns = {'mapstd'};

    % Prepare the data for training and simulation
    [x,xi,ai,t] = preparets(net,X,{},T);

    % Setup division of data for training, validation, testing
   
    % Setup initial weights
    net = configure(net,x,t);
    net.IW{1,1} = net_org.IW{1,1};
    net.IW{1,2} = net_org.IW{1,2};
    net.LW{2,1} = net_org.LW{2,1};
    
    % Test the network and compute metrics
    y = net_org(x,xi,ai);
    e = gsubtract(t,y);
    e = cell2mat(e);
    a =  mse(e);
    % Closed loop network
    netc = closeloop(net_org);
    view(netc)
    [xc,xic,aic,~] = preparets(netc,X,{},T);
    yc = netc(xc,xic,aic);  
    
    
    figure
    plot(1:length(cell2mat(T)),cell2mat(T))
    hold on
    plot(3:length(cell2mat(T)), cell2mat(y))
    plot(3:length(cell2mat(T)), cell2mat(yc))
    title("Monthly Milk Production")
    xlabel("Months enconded")
    ylabel("Milk production (pounds per cow)")
    legend('Original','Open Loop', 'Close Loop', 'Location','northwest')
    hold off

end