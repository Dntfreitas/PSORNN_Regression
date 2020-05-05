%% Time-Series

% Traffic
disp("Traffic");
vars = [3, 10, 12, 14, 16];
nInputs = length(vars);
data = traffic(:, [vars, end]);    
xLabel = 'Hour of day';
yLabel = 'Slowness in traffic (%)';
pltTitle = 'Urban Traffic in São Paulo - Brazil';
name = 'traffic';
[~, ~, ~] = tmsr(data, nInputs, xLabel, yLabel, pltTitle, name);

% Milk
disp("Milk");
vars = [1];
nInputs = length(vars);
data = milk(:, [vars, end]);
xLabel = 'Month';
yLabel = 'Pounds per cow';
pltTitle = 'Monthly Milk Production';
name = 'milk';
[~, ~, ~] = tmsr(data, nInputs, xLabel, yLabel, pltTitle, name);

%% Regression

% Fish
disp("Fish");
vars = [1, 2, 3, 6];
nInputs = length(vars);
data = fish(:, [vars, end]);
name = 'fish';
[~, ~, ~] = regr(data, nInputs, name);

% Red wine
disp("Red Wine");
vars = [1, 2, 3, 5, 7, 8, 10, 11];
nInputs = length(vars);
data = winered(:, [vars, end]);
name = 'redWine';
[~, ~, ~] = regr(data, nInputs, name);

% White wine
disp("White Wine");
vars = [3, 5, 8, 11];
nInputs = length(vars);
data = winewhite(:, [vars, end]);
name = 'whiteWine';
[~, ~, ~] = regr(data, nInputs, name);
