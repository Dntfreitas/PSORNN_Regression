%% Fish 
inputs = fish(:, 1:6);
outputs = fish(:,7);

[~,scores] = fsrftest(inputs, outputs);
figure
bar(scores)
title("Univariate feature selection")
xlabel("Features")
ylabel("Scores")

% 1) CIC0
% 2) SM1_Dz(Z)
% 3) GATS1i
% 6) MLOGP

%% Traffic
inputs = traffic(:, 1:17);
outputs = traffic(:, end);

[~,scores] = fsrftest(inputs, outputs);
figure
bar(scores)
title("Univariate feature selection")
xlabel("Features")
ylabel("Scores")

% 3) Broken Truck
% 10) Lack of electricity
% 12) Point of flooding 
% 14) Defect in the network of trolleybuses
% 16) Semaphore off

%% Milk
inputs = milk(:, 1:2);
outputs = milk(:, end);

[~,scores] = fsrftest(inputs, outputs);
figure
bar(scores)
title("Univariate feature selection")
xlabel("Features")
ylabel("Scores")

% 1) Year

%% Red wine
inputs = winered(:, 1:11);
outputs = winered(:, end);

[~,scores] = fsrftest(inputs, outputs);
figure
bar(scores)
title("Univariate feature selection")
xlabel("Features")
ylabel("Scores")

% 1) Fixed acidity
% 2) Volatile acidity
% 3) Citric acid
% 5) Chlorides
% 7) Total sulfur dioxide
% 8) Density
% 10) Sulphates
% 11) Alcohol

%% White wine
inputs = winewhite(:, 1:11);
outputs = winewhite(:, end);

[~,scores] = fsrftest(inputs, outputs);
figure
bar(scores)
title("Univariate feature selection")
xlabel("Features")
ylabel("Scores")

% 3) Citric acid
% 5) Chlorides
% 8) Density
% 11) Alcohol