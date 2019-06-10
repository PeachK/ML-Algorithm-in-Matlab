function  s=K(x1,x2,type,p,sigma)
%K 
%   此处显示详细说明
    switch type
        case '1' 
             s=x1*x2';
        case 'poly'
             s=(x1*x2'+1)^p;
        case 'Gauss'
             s=exp(-sqrt((x1-x2)*(x1-x2))/(2*sigma)^2);
    end
end


