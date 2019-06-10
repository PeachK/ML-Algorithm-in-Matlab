clear ;
close all;
clc;
t1  = 5+4*randn(2,10);
t2 = 20+4*randn(2,10);

X = [t1 t2]';% 20*2    20��2ά����
Y = [ones(1,10)  zeros(1,10)]; %1*20  ǰʮ�����·��ĵ���1

%��ʼ��Ȩֵ����������������Ԫ,�������һ����Ԫ����������ǣ�1������ 0�����ϣ�
[row,col]=size(X);
[SamIn,minp,maxp,tn,mint,maxt]=premnmx(p,t); %ԭʼ�����ԣ�������������һ��

rand('state',sum(100*clock))   %����ϵͳʱ�����Ӳ��������         
NoiseVar=0.01;                    %����ǿ��Ϊ0.01�����������Ŀ����Ϊ�˷�ֹ���������ϣ�
Noise=NoiseVar*randn(2,SamNum);   %��������
SamOut=tn + Noise;                   %��������ӵ����������

W1=0.5*rand(3,col)-0.1;   %��ʼ���������������֮���Ȩֵ
b1=0.5*rand(3,1)-0.1;       %��ʼ���������������֮�����ֵ
W2=0.5*rand(1,3)-0.1; %��ʼ���������������֮���Ȩֵ              
b2=0.5*rand(1,1)-0.1;                %��ʼ���������������֮�����ֵ
eta=0.002; %ѧϰ��
e0=0.001;%���


%ʹ�����ݽ���ѵ��,�����ز�ļ������logsig(x)��f(x)=1/(1+exp(-x));
for j=1:5000%����������

  
    HiddenOut=logsig(W1*X'+repmat(b1,1,row));%���ز����� 3*20
    BpOut=logsig(W2*HiddenOut+repmat(b2,1,row));%1*20
    error=Y- BpOut;%1*20
    E=sumsqr(error);%sumsqr���������ƽ����
    if E<e0   %���ȴ�������ѵ��
        break;
    end
    %����Ȩֵ����ֵ
    Delta2=error;%1*20
    Delta1=W2'*Delta2.*HiddenOut.*(1-HiddenOut);  %3*20  

    dW2=eta*Delta2*HiddenOut';
    dB2=eta*Delta2*ones(row,1);
    
    dW1=eta*Delta1*X;
    dB1=eta*Delta1*ones(row,1);%3*1
    %���������������֮���Ȩֵ����ֵ��������
    W2=W2+dW2;
    b2=b2+dB2;
    %���������������֮���Ȩֵ����ֵ��������
    W1=W1+dW1;
    b1=b1+dB1;
end 
    

t1  = 5+4*randn(2,10);
t2 = 20+4*randn(2,10);

test_X = [t1 t2]';% 20*2    20��2ά����
test_Y = [ones(10,1)  zeros(10,1)]; %1*20  ǰʮ�����·��ĵ���1    
plot(test_X(:,1),test_X(:,2),'bx');

hold on;
%����һ��   
[test_row,test_col]=size(test_X);
HiddenOut=logsig(W1*test_X'+repmat(b1,1,test_row)); % ������������ս��
NetworkOut=W2*HiddenOut+repmat(b2,1,test_row);    % �����������ս��
NetworkOut

    
% premnmx��tramnmx��postmnmx��mapminmax
% premnmx�������ڽ�������������ݻ�������ݽ��й�һ������һ��������ݽ��ֲ���[-1,1]�����ڡ�
% premnmx�����﷨��ʽ�ǣ�[Pn,minp,maxp,Tn,mint,maxt]=premnmx(P,T)������P��T�ֱ�Ϊԭʼ�����������ݡ�
% ��ѵ������ʱ������õ��Ǿ�����һ�����������ݣ���ô�Ժ�ʹ������ʱ���õ�������ҲӦ�ú��������ݽ�����ͬ��Ԥ�������Ҫ�õ�tramnmx������
% tramnmx�����﷨��ʽ�ǣ�[PN]=tramnmx(P,minp,maxp)
% ����P��PN�ֱ�Ϊ�任ǰ������������ݣ�maxp��minp�ֱ�Ϊpremnmx�����ҵ������ֵ����Сֵ��
% ������������Ҫ���з���һ����ԭ��ԭʼ�����ݣ����õĺ����ǣ�postmnmx��
% postmnmx�����﷨��ʽ�ǣ�[PN] = postmnmx(P,minp,maxp)
% ����P��PN�ֱ�Ϊ�任ǰ������������ݣ�maxp��minp�ֱ�Ϊpremnmx�����ҵ������ֵ����Сֵ��
% ����һ��������mapminmax���ú������԰Ѿ����ÿһ�й�һ��[-1 1].
% mapminmax�����﷨��ʽ�ǣ�[y1,PS] = mapminmax(x1)
% ����x1 ����Ҫ��һ�ľ��� y1�ǽ����
% ����Ҫ������һ����������һʱ���Ϳ���������ķ�������ͬ�Ĺ�һ��
% y2 = mapminmax('apply',x2,PS)
% ����Ҫ�ѹ�һ�����ݻ�ԭʱ���������������
% x1_again = mapminmax('reverse',y1,PS)
% prestd��poststd��trastd
% prestd��һ����λ��������ֵ��
% pminp��maxp�ֱ�ΪP�е���Сֵ�����ֵ��mint��maxt�ֱ�ΪT����Сֵ�����ֵ��    