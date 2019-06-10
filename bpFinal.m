function bpFinal(X,Y,UnitNum,func)
%BPFINAL �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
[InDim,~]=size(X); %����ά��   
[OutDim,SamNum]=size(Y);% ���ά��   ������
HiddenUnitNum=UnitNum;%���ز���Ԫ����

[X_in,min_x,max_x,Y_in,min_y,max_y]=premnmx(X,Y); %ԭʼ�����ԣ�������������һ��


rand('state',sum(100*clock))   %����ϵͳʱ�����Ӳ��������         
NoiseVar=0.01;                    %����ǿ��Ϊ0.01�����������Ŀ����Ϊ�˷�ֹ���������ϣ�
Noise=NoiseVar*randn(OutDim,SamNum);   %��������
Y_out=Y_in + Noise;                   %��������ӵ����������

Epochs=400000;                              %���ѵ������Ϊ50000
eta=0.00001;                                       %ѧϰ����Ϊ0.035
E0=0.0005;                              %Ŀ�����Ϊ0.0005
W1=0.5*rand(HiddenUnitNum,InDim)-0.1;   %��ʼ���������������֮���Ȩֵ
B1=0.5*rand(HiddenUnitNum,1)-0.1;       %��ʼ���������������֮�����ֵ
W2=0.5*rand(OutDim,HiddenUnitNum)-0.1; %��ʼ���������������֮���Ȩֵ              
B2=0.5*rand(OutDim,1)-0.1;    %��ʼ���������������֮�����ֵ



ErrHistory=[];  %���м����Ԥ��ռ���ڴ�
for i=1:Epochs
    temp=(W1*X_in+repmat(B1,1,SamNum));
    HiddenOut=AcFun(temp,func); % �������������
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
    switch func
        case 'sigmoid'
              Delta1=W2'*Delta2.*HiddenOut.*(1-HiddenOut);    
        case 'tanh'
              Delta1=W2'*Delta2.*(1-HiddenOut.^2);  
        case 'Relu'
              if temp>0
                 Delta1=W2'*Delta2;
              else
                 Delta1=W2'*Delta2.*0;
              end
    end
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
HiddenOut=AcFun((W1*X_in+repmat(B1,1,SamNum)),func); % ������������ս��
NetworkOut=W2*HiddenOut+repmat(B2,1,SamNum);    % �����������ս��

a=postmnmx(NetworkOut,min_y,max_y);               % ��ԭ���������Ľ��

figure ;
subplot(2,1,2);
plot(1:SamNum,Y,'r-o',1:SamNum,a,'b--x');
title('Ԥ��ֵ��ʵ��ֵ�Ա�');
legend('ʵ�ʱ��','Ԥ��ı��');
xlabel('���');ylabel('���ֵ��0��1��');
subplot(2,1,1);
[~,Ey]=size(ErrHistory);
plot(1:100:Ey,ErrHistory(1:100:Ey),'ro');
title('���ֵ��ѵ����������������');
xlabel('ѵ������');ylabel('���ֵ');
ErrSum=0;
for i=1:SamNum
    ErrSum=ErrSum+(Y(i)-a(i))^2;
end
    fprintf('�����%5f\n',ErrSum/SamNum);
end

