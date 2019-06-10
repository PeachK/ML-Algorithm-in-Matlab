clear ;
close all;
clc;
t1  = 5+4*randn(2,10);
t2 = 20+4*randn(2,10);

X = [t1 t2]';% 20*2    20个2维数据
Y = [ones(1,10)  zeros(1,10)]; %1*20  前十个在下方的点标记1

%初始化权值，设隐层有三个神经元,输出层有一个神经元，最后输出标记（1在右下 0在左上）
[row,col]=size(X);
[SamIn,minp,maxp,tn,mint,maxt]=premnmx(p,t); %原始样本对（输入和输出）归一化

rand('state',sum(100*clock))   %依据系统时钟种子产生随机数         
NoiseVar=0.01;                    %噪声强度为0.01（添加噪声的目的是为了防止网络过度拟合）
Noise=NoiseVar*randn(2,SamNum);   %生成噪声
SamOut=tn + Noise;                   %将噪声添加到输出样本上

W1=0.5*rand(3,col)-0.1;   %初始化输入层与隐含层之间的权值
b1=0.5*rand(3,1)-0.1;       %初始化输入层与隐含层之间的阈值
W2=0.5*rand(1,3)-0.1; %初始化输出层与隐含层之间的权值              
b2=0.5*rand(1,1)-0.1;                %初始化输出层与隐含层之间的阈值
eta=0.002; %学习率
e0=0.001;%误差


%使用数据进行训练,设隐藏层的激活函数是logsig(x)：f(x)=1/(1+exp(-x));
for j=1:5000%可设置轮数

  
    HiddenOut=logsig(W1*X'+repmat(b1,1,row));%隐藏层的输出 3*20
    BpOut=logsig(W2*HiddenOut+repmat(b2,1,row));%1*20
    error=Y- BpOut;%1*20
    E=sumsqr(error);%sumsqr函数求矩阵平方和
    if E<e0   %精度达标则结束训练
        break;
    end
    %更新权值和阈值
    Delta2=error;%1*20
    Delta1=W2'*Delta2.*HiddenOut.*(1-HiddenOut);  %3*20  

    dW2=eta*Delta2*HiddenOut';
    dB2=eta*Delta2*ones(row,1);
    
    dW1=eta*Delta1*X;
    dB1=eta*Delta1*ones(row,1);%3*1
    %对输出层与隐含层之间的权值和阈值进行修正
    W2=W2+dW2;
    b2=b2+dB2;
    %对输入层与隐含层之间的权值和阈值进行修正
    W1=W1+dW1;
    b1=b1+dB1;
end 
    

t1  = 5+4*randn(2,10);
t2 = 20+4*randn(2,10);

test_X = [t1 t2]';% 20*2    20个2维数据
test_Y = [ones(10,1)  zeros(10,1)]; %1*20  前十个在下方的点标记1    
plot(test_X(:,1),test_X(:,2),'bx');

hold on;
%测试一下   
[test_row,test_col]=size(test_X);
HiddenOut=logsig(W1*test_X'+repmat(b1,1,test_row)); % 隐含层输出最终结果
NetworkOut=W2*HiddenOut+repmat(b2,1,test_row);    % 输出层输出最终结果
NetworkOut

    
% premnmx、tramnmx、postmnmx、mapminmax
% premnmx函数用于将网络的输入数据或输出数据进行归一化，归一化后的数据将分布在[-1,1]区间内。
% premnmx语句的语法格式是：[Pn,minp,maxp,Tn,mint,maxt]=premnmx(P,T)，其中P，T分别为原始输入和输出数据。
% 在训练网络时如果所用的是经过归一化的样本数据，那么以后使用网络时所用的新数据也应该和样本数据接受相同的预处理，这就要用到tramnmx函数：
% tramnmx语句的语法格式是：[PN]=tramnmx(P,minp,maxp)
% 其中P和PN分别为变换前、后的输入数据，maxp和minp分别为premnmx函数找到的最大值和最小值。
% 网络输出结果需要进行反归一化还原成原始的数据，常用的函数是：postmnmx。
% postmnmx语句的语法格式是：[PN] = postmnmx(P,minp,maxp)
% 其中P和PN分别为变换前、后的输入数据，maxp和minp分别为premnmx函数找到的最大值和最小值。
% 还有一个函数是mapminmax，该函数可以把矩阵的每一行归一到[-1 1].
% mapminmax语句的语法格式是：[y1,PS] = mapminmax(x1)
% 其中x1 是需要归一的矩阵 y1是结果。
% 当需要对另外一组数据做归一时，就可以用下面的方法做相同的归一了
% y2 = mapminmax('apply',x2,PS)
% 当需要把归一的数据还原时，可以用以下命令：
% x1_again = mapminmax('reverse',y1,PS)
% prestd、poststd、trastd
% prestd归一到单位方差和零均值。
% pminp和maxp分别为P中的最小值和最大值。mint和maxt分别为T的最小值和最大值。    