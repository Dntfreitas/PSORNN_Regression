function [w] = initWeigths(nInputs, sz1, sz2)
    l = 2.4 * nInputs;
    a = -l;
    b = l;
    w = (b-a).*rand(sz1, sz2) + a;
end