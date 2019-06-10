function main()
clc                          % 清屏
clear all;                  %清除内存以便加快运算速度
close all;                  %关闭当前所有figure图像
SamNum=20;                  %输入样本数量为20
TestSamNum=20;              %测试样本数量也是20
ForcastSamNum=20;            %预测样本数量为2
HiddenUnitNum=8;            %中间层隐节点数量取8,比工具箱程序多了1个
InDim=2;                    %网络输入维度为2
OutDim=1;                   %网络输出维度为1

%训练数据
t1  = 5+4*randn(2,10);
t2 = 20+4*randn(2,10);
X= [t1 t2];% 2*20    20个2维数据
Y = [ones(1,10)  zeros(1,10)]; %1*20  前十个在下方的点标记1

[X_in,min_x,max_x,Y_in,min_y,max_y]=premnmx(X,Y); %原始样本对（输入和输出）归一化

rand('state',sum(100*clock))   %依据系统时钟种子产生随机数         
NoiseVar=0.01;                    %噪声强度为0.01（添加噪声的目的是为了防止网络过度拟合）
Noise=NoiseVar*randn(1,SamNum);   %生成噪声
Y_out=Y_in + Noise;                   %将噪声添加到输出样本上

TestSamIn=X_in;                           %这里取输入样本与测试样本相同因为样本容量偏少
TestSamOut=Y_out;%Samout                         %也取输出样本与测试样本相同

Epochs=50000;                              %最多训练次数为50000
eta=0.035;                                       %学习速率为0.035
E0=0.0005;                              %目标误差为0.0005
W1=0.5*rand(HiddenUnitNum,InDim)-0.1;   %初始化输入层与隐含层之间的权值
B1=0.5*rand(HiddenUnitNum,1)-0.1;       %初始化输入层与隐含层之间的阈值
W2=0.5*rand(OutDim,HiddenUnitNum)-0.1; %初始化输出层与隐含层之间的权值              
B2=0.5*rand(OutDim,1)-0.1;                %初始化输出层与隐含层之间的阈值

ErrHistory=[];  %给中间变量预先占据内存
for i=1:Epochs
    
    HiddenOut=logsig(W1*X_in+repmat(B1,1,SamNum)); % 隐含层网络输出
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
    Delta1=W2'*Delta2.*HiddenOut.*(1-HiddenOut);    

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

HiddenOut=logsig(W1*X_in+repmat(B1,1,TestSamNum)); % 隐含层输出最终结果
NetworkOut=W2*HiddenOut+repmat(B2,1,TestSamNum);    % 输出层输出最终结果

a=postmnmx(NetworkOut,min_y,max_y);               % 还原网络输出层的结果


% 利用训练好的网络进行预测
% 当用训练好的网络对新数据进行预测时，先进行同样的归一化处理
tnew1  = 5+4*randn(2,10);
tnew2 = 20+4*randn(2,10);
Xnew= [tnew1 tnew2];% 2*20    20个2维数据
Ynew = [ones(1,10)  zeros(1,10)]; %1*20  前十个在下方的点标记1

Xnn=tramnmx(Xnew,min_x,max_x);         %利用原始输入数据的归一化参数对新数据进行归一化；
HiddenOut=logsig(W1*Xnn+repmat(B1,1,ForcastSamNum)); % 隐含层输出预测结果
ynewn=W2*HiddenOut+repmat(B2,1,ForcastSamNum)           % 输出层输出预测结果

%把网络预测得到的数据还原为原始的数量级；
ynew=postmnmx(ynewn,min_y,max_y)
figure ;
subplot(2,1,2);
plot(1:20,Ynew,'r-o',1:20,ynew,'b--x');
title('预测值和实际值对比');
legend('实际标记','预测的标记');
xlabel('点号');ylabel('标记值（0或1）');
subplot(2,1,1);
plot(1:100,ErrHistory(1:100),'ro');
title('误差值随训练轮数的收敛过程');
xlabel('训练轮数');ylabel('误差值');










