function bpFinal(X,Y,UnitNum,func)
%BPFINAL 此处显示有关此函数的摘要
%   此处显示详细说明
[InDim,~]=size(X); %输入维度   
[OutDim,SamNum]=size(Y);% 输出维度   样本数
HiddenUnitNum=UnitNum;%隐藏层神经元个数

[X_in,min_x,max_x,Y_in,min_y,max_y]=premnmx(X,Y); %原始样本对（输入和输出）归一化


rand('state',sum(100*clock))   %依据系统时钟种子产生随机数         
NoiseVar=0.01;                    %噪声强度为0.01（添加噪声的目的是为了防止网络过度拟合）
Noise=NoiseVar*randn(OutDim,SamNum);   %生成噪声
Y_out=Y_in + Noise;                   %将噪声添加到输出样本上

Epochs=400000;                              %最多训练次数为50000
eta=0.00001;                                       %学习速率为0.035
E0=0.0005;                              %目标误差为0.0005
W1=0.5*rand(HiddenUnitNum,InDim)-0.1;   %初始化输入层与隐含层之间的权值
B1=0.5*rand(HiddenUnitNum,1)-0.1;       %初始化输入层与隐含层之间的阈值
W2=0.5*rand(OutDim,HiddenUnitNum)-0.1; %初始化输出层与隐含层之间的权值              
B2=0.5*rand(OutDim,1)-0.1;    %初始化输出层与隐含层之间的阈值



ErrHistory=[];  %给中间变量预先占据内存
for i=1:Epochs
    temp=(W1*X_in+repmat(B1,1,SamNum));
    HiddenOut=AcFun(temp,func); % 隐含层网络输出
    NetworkOut=W2*HiddenOut+repmat(B2,1,SamNum);    % 输出层网络输出
    Error=Y_out-NetworkOut;      % 实际输出与网络输出之差
    SSE=sumsqr(Error);               %误差平方和

    ErrHistory=[ErrHistory SSE];

    if SSE<E0
        break;
    end      %如果达到误差要求则跳出学习循环
    
    % 以下六行是BP网络最核心的程序
    % 他们是权值（阈值）依据能量函数负梯度下降原理所作的每一步动态调整量
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
    %对输出层与隐含层之间的权值和阈值进行修正
    W2=W2+eta*dW2;
    B2=B2+eta*dB2;
    %对输入层与隐含层之间的权值和阈值进行修正
    W1=W1+eta*dW1;
    B1=B1+eta*dB1;
end
HiddenOut=AcFun((W1*X_in+repmat(B1,1,SamNum)),func); % 隐含层输出最终结果
NetworkOut=W2*HiddenOut+repmat(B2,1,SamNum);    % 输出层输出最终结果

a=postmnmx(NetworkOut,min_y,max_y);               % 还原网络输出层的结果

figure ;
subplot(2,1,2);
plot(1:SamNum,Y,'r-o',1:SamNum,a,'b--x');
title('预测值和实际值对比');
legend('实际标记','预测的标记');
xlabel('点号');ylabel('标记值（0或1）');
subplot(2,1,1);
[~,Ey]=size(ErrHistory);
plot(1:100:Ey,ErrHistory(1:100:Ey),'ro');
title('误差值随训练轮数的收敛过程');
xlabel('训练轮数');ylabel('误差值');
ErrSum=0;
for i=1:SamNum
    ErrSum=ErrSum+(Y(i)-a(i))^2;
end
    fprintf('误差是%5f\n',ErrSum/SamNum);
end

