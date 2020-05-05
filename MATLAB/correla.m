[l,c] = size(dataset);
nvar = c - 1;

target = dataset(:,end);
predictors = dataset(:,1:nvar);


V = zeros(5, nvar);

V(1,:) = var(predictors);
V(2,:) = corr(predictors, target);
