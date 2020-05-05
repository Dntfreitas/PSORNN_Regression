function [t] = mmovw(y, ws, wa)
    
    n = length(y);
    q = length(ws);
    d =  floor(q/2);
    t = zeros(n,1);
    
    t_aux = conv(y, ws, 'same');
    t(d+1:n-d) = t_aux(d+1:n-d);
    
    for i = 1:d
       for j = 1:q
           t(i) =  t(i) + wa(i, j) * y(j);
           t(end-i+1) = t(end-i+1) + wa(i, j) * y(end-j+1);
       end
    end
end