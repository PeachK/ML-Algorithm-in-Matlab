function  s= AcFun(X,func)
%ACFUN 此处显示有关此函数的摘要
%   此处显示详细说明
switch func
    case 'sigmoid'
        s=logsig(X);
    case 'tanh'
        s=tanh(X);
    case 'Relu'
        s=max(0,X);
end
end

