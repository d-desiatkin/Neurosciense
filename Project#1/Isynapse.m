function [I] = Isynapse(v,i)
%ISYN Summary of this function goes here
%   Detailed explanation goes here
I = 0;
global ge;
for j = 1:length(v)
    if i ~= j
        I = I + ge*(v(j)-v(i));
    end
end

end

