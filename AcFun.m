function  s= AcFun(X,func)
%ACFUN �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
switch func
    case 'sigmoid'
        s=logsig(X);
    case 'tanh'
        s=tanh(X);
    case 'Relu'
        s=max(0,X);
end
end

