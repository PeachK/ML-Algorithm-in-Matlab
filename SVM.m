% t1  = 5+4*randn(2,10);
% t2 = 20+4*randn(2,10);
% 
% X = [t1 t2]';
% Y = [ones(10,1) ; -ones(10,1)];
% 
% plot(X(1:10,1),X(1:10,2),'ro');
% grid on;
% hold on;
% plot(X(11:20,1),X(11:20,2),'bx');
[heart_scale_label,heart_scale_inst]=libsvmread('heart_scale');
X=full(heart_scale_inst);
Y=heart_scale_label;


w=0;
b=0;
C=0.4;
e=0.001;
[row,col]=size(X);
MAX=row;
iter=0;

a=zeros(row,1);

while(iter<=MAX)
    iterchange=0;
   
    for i=1:row
        %fxi=double((a.*Y)'*(X*X(i,:)'))+b;
        fxi=double((a.*Y)'*K(X,X(i,:),'1',1,2))+b;
        ei=fxi-double(Y(i));
        
        if(a(i)<C&&Y(i)*ei<-e)||(a(i)>0&&Y(i)*ei>e)
              j=i;
              while j==i
                  j=randi(row,1,1);
              end
              %fxj=double((a.*Y)'*(X*X(j,:)'))+b;
              fxj=double((a.*Y)'*K(X,X(j,:),'1',1,2))+b;
              ej=fxj-double(Y(j));
              aiold=a(i);
              ajold=a(j);
              if(Y(j)~=Y(i))
                  L=max(0,a(j)-a(i));
                  H=min(C,C+a(j)-a(i));
              else
                  L=max(0,a(j)+a(i)-C);
                  H=min(C,a(j)+a(i));
              end
              if L==H
                  continue;
              end
              kii=K(X(i,:),X(i,:),'1',1,2);
              kjj=K(X(j,:),X(j,:),'1',1,2);
              kij=K(X(i,:),X(j,:),'1',1,2);
              eta=kii+kjj-2*kij;
              if(eta<=0)
                  continue;
              end
              a(j)=a(j)+Y(j)*(ei-ej)/eta;% aj new unclipped
              if a(j)>H
                  a(j)=H;
              end
              if a(j)<L
                  a(j)=L;
              end
              if abs(a(j)-ajold)<0.00001
                  continue;
              end
              a(i)=a(i)+Y(i)*Y(j)*(ajold-a(j));
              b1=b-ei-Y(i)*(a(i)-aiold)*kii-Y(j)*kij*(a(j)-ajold);
              b2=b-ej-Y(i)*(a(i)-aiold)*kij-Y(j)*kjj*(a(j)-ajold);
              if a(i)>0&&a(i)<C
                  b=b1;
              elseif a(j)>0&&a(j)<C
                  b=b2;
              else
                  b=(b1+b2)/2;
              end
              iterchange=iterchange+1;
        end
    end
    if iterchange==0
        iter=iter+1;
    else
        iter=0;
    end
end
w=zeros(1,col);% 1*clµÄ¾ØÕó
for i=1:row
    w=w+a(i)*Y(i)*X(i,:);
%     if(a(i)~=0)
%         plot(X(i,1),X(i,2),'gs');
%     end
end
count=0;
for i=1:row
    fxi=w*X(i,:)'+b;
    if (Y(i)==-1&&fxi<0)
        count=count+1;
    elseif (Y(i)==1&&fxi>0)
        count=count+1;
    end
end
fprintf('the accuracy is %.3f(%d/%d)\n',count/row,count,row);
% x1=0:0.1:20;
% x2=(-w(1)*x1-b)/w(2);
% plot(x1,x2,'r');
% f=Y(i)*(w*X'+b);