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