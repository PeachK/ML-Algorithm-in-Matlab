function main()
clc                          % ����
clear all;                  %����ڴ��Ա�ӿ������ٶ�
close all;                  %�رյ�ǰ����figureͼ��
SamNum=20;                  %������������Ϊ20
TestSamNum=20;              %������������Ҳ��20
ForcastSamNum=20;            %Ԥ����������Ϊ2
HiddenUnitNum=8;            %�м�����ڵ�����ȡ8,�ȹ�����������1��
InDim=2;                    %��������ά��Ϊ2
OutDim=1;                   %�������ά��Ϊ1

%ѵ������
t1  = 5+4*randn(2,10);
t2 = 20+4*randn(2,10);
X= [t1 t2];% 2*20    20��2ά����
Y = [ones(1,10)  zeros(1,10)]; %1*20  ǰʮ�����·��ĵ���1

[X_in,min_x,max_x,Y_in,min_y,max_y]=premnmx(X,Y); %ԭʼ�����ԣ�������������һ��

rand('state',sum(100*clock))   %����ϵͳʱ�����Ӳ��������         
NoiseVar=0.01;                    %����ǿ��Ϊ0.01�����������Ŀ����Ϊ�˷�ֹ���������ϣ�
Noise=NoiseVar*randn(1,SamNum);   %��������
Y_out=Y_in + Noise;                   %��������ӵ����������

TestSamIn=X_in;                           %����ȡ�������������������ͬ��Ϊ��������ƫ��
TestSamOut=Y_out;%Samout                         %Ҳȡ������������������ͬ

Epochs=50000;                              %���ѵ������Ϊ50000
eta=0.035;                                       %ѧϰ����Ϊ0.035
E0=0.0005;                              %Ŀ�����Ϊ0.0005
W1=0.5*rand(HiddenUnitNum,InDim)-0.1;   %��ʼ���������������֮���Ȩֵ
B1=0.5*rand(HiddenUnitNum,1)-0.1;       %��ʼ���������������֮�����ֵ
W2=0.5*rand(OutDim,HiddenUnitNum)-0.1; %��ʼ���������������֮���Ȩֵ              
B2=0.5*rand(OutDim,1)-0.1;                %��ʼ���������������֮�����ֵ

ErrHistory=[];  %���м����Ԥ��ռ���ڴ�
for i=1:Epochs
    
    HiddenOut=logsig(W1*X_in+repmat(B1,1,SamNum)); % �������������
    NetworkOut=W2*HiddenOut+repmat(B2,1,SamNum);    % ������������
    Error=Y_out-NetworkOut;      % ʵ��������������֮��
    SSE=sumsqr(Error);               %���ƽ����

    ErrHistory=[ErrHistory SSE];

    if SSE<E0
        break;
    end      %����ﵽ���Ҫ��������ѧϰѭ��
    
    % ����������BP��������ĵĳ���
    % ������Ȩֵ����ֵ�����������������ݶ��½�ԭ��������ÿһ����̬������
    Delta2=Error;
    Delta1=W2'*Delta2.*HiddenOut.*(1-HiddenOut);    

    dW2=Delta2*HiddenOut';
    dB2=Delta2*ones(SamNum,1);
    
    dW1=Delta1*X_in';
    dB1=Delta1*ones(SamNum,1);
    %���������������֮���Ȩֵ����ֵ��������
    W2=W2+eta*dW2;
    B2=B2+eta*dB2;
    %���������������֮���Ȩֵ����ֵ��������
    W1=W1+eta*dW1;
    B1=B1+eta*dB1;
end

HiddenOut=logsig(W1*X_in+repmat(B1,1,TestSamNum)); % ������������ս��
NetworkOut=W2*HiddenOut+repmat(B2,1,TestSamNum);    % �����������ս��

a=postmnmx(NetworkOut,min_y,max_y);               % ��ԭ���������Ľ��


% ����ѵ���õ��������Ԥ��
% ����ѵ���õ�����������ݽ���Ԥ��ʱ���Ƚ���ͬ���Ĺ�һ������
tnew1  = 5+4*randn(2,10);
tnew2 = 20+4*randn(2,10);
Xnew= [tnew1 tnew2];% 2*20    20��2ά����
Ynew = [ones(1,10)  zeros(1,10)]; %1*20  ǰʮ�����·��ĵ���1

Xnn=tramnmx(Xnew,min_x,max_x);         %����ԭʼ�������ݵĹ�һ�������������ݽ��й�һ����
HiddenOut=logsig(W1*Xnn+repmat(B1,1,ForcastSamNum)); % ���������Ԥ����
ynewn=W2*HiddenOut+repmat(B2,1,ForcastSamNum)           % ��������Ԥ����

%������Ԥ��õ������ݻ�ԭΪԭʼ����������
ynew=postmnmx(ynewn,min_y,max_y)
figure ;
subplot(2,1,2);
plot(1:20,Ynew,'r-o',1:20,ynew,'b--x');
title('Ԥ��ֵ��ʵ��ֵ�Ա�');
legend('ʵ�ʱ��','Ԥ��ı��');
xlabel('���');ylabel('���ֵ��0��1��');
subplot(2,1,1);
plot(1:100,ErrHistory(1:100),'ro');
title('���ֵ��ѵ����������������');
xlabel('ѵ������');ylabel('���ֵ');










