clear
clc

%% Brazil Dataset

brazil_data=readtable('brazil.csv'); %Load the excel file

% Splitting the data in to two time series X(t) and Y(t) as needed by NARX model
X_brazil=brazil_data(:,1) % X(t) Extraction
X_brazil=X_brazil{:,:}; % Changing the table format to matrix for X(t)
X_brazil=X_brazil-X_brazil(1); % Creating the reference point for X(t)
X_brazil=rescale(X_brazil); % feature scaling X(t)

Y_brazil=brazil_data(:,end);% Y(t) Extraction
Y_brazil=str2double(Y_brazil{:,:});% Changing the table format to matrix for Y(t)
Y_brazil=rescale(Y_brazil);% feature scaling Y(t)

%% Milk Production
milk_prod=readtable('monthly-milk-production-pounds-p.csv'); %Load the excel file

% Splitting the data in to two time series X(t) and Y(t) as needed by NARX model
milk_prod=milk_prod(1:end-1,:);
% X(t) Extraction
X_milk=milk_prod(:,1);
X_milk=X_milk{:,:};

X_milk=datenum(X_milk)*30; % Date Preprocessing
X_milk=rescale(X_milk); % feature scaling X(t)

Y_milk=milk_prod(:,2); % Y(t) Extraction
Y_milk=Y_milk{:,:};% Changing the table format to matrix for Y(t)
Y_milk=rescale(Y_milk); % feature scaling Y(t)
%% Wine Quality
wine_quality_red=readtable('winequality-red.csv');  %Load the excel file

% Splitting the data into dependent(Y) and independent(X) 
X_wine_red=wine_quality_red(:,1:end-1);% X Extraction
Y_wine_red=wine_quality_red(:,end);% Y Extraction
X_wine_red=X_wine_red{:,:};% Changing the table format to matrix for X
Y_wine_red=Y_wine_red{:,:};% Changing the table format to matrix for Y


%white wine
wine_quality_white=readtable('winequality-white.csv'); %Load the excel file
% Splitting the data into dependent(Y) and independent(X)
X_wine_white=wine_quality_white(:,1:end-1);% X Extraction
Y_wine_white=wine_quality_white(:,end);% Y Extraction
X_wine_white=X_wine_white{:,:};% Changing the table format to matrix for X
Y_wine_white=Y_wine_white{:,:};% Changing the table format to matrix for Y
%% qsar fish toxicity
qsar_fish_tox=readtable('qsar_fish_toxicity.csv');%Load the excel file
% Splitting the data into dependent(Y) and independent(X)
X_qsar=qsar_fish_tox(:,1:end-1);% X Extraction
Y_qsar=qsar_fish_tox(:,1:end);% Y Extraction
X_qsar=X_qsar{:,:};% Changing the table format to matrix for X
Y_qsar=Y_qsar{:,:};% Changing the table format to matrix for Y



%% Time-Series

nInputs = 17;
xLabel = 'Hour of day';
yLabel = 'Slowness in traffic (%)';
pltTitle = 'Urban Traffic in São Paulo - Brazil';
name = 'traffic';
[r2, bestNet2, worstNet2] = tmsr(traffic, nInputs, xLabel, yLabel, pltTitle, name);


nInputs = 2;
xLabel = 'Month';
yLabel = 'Pounds per cow';
pltTitle = 'Monthly Milk Production';
name = 'milk';
[r1, bestNet1, worstNet1] = tmsr(milk, nInputs, xLabel, yLabel, pltTitle, name);

%% Regression
nInputs = 6;
name = 'fish';
[r1, bestNet1, worstNet1] = regr(fish, nInputs, name);

nInputs = 11;
name = 'redWine';
[r2, bestNet2, worstNet2] = regr(winered, nInputs, name);

nInputs = 11;
name = 'whiteWine';
[r3, bestNet3, worstNet3] = regr(winewhite, nInputs, name);
