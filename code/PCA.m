% PCA demo for data warehouse and data mining in bupt.
% Feature Matrix cx. Each colum represents a feature and each row a sample
% data
cx = [ 1.4000 1.55000
       3.0000 3.2000
       0.6000 0.7000
       2.2000 2.3000
       1.8000 2.1000
       2.0000 1.6000
       1.0000 1.1000
       2.5000 2.4000
       1.5000 1.6000
       1.2000 0.8000
       2.1000 2.5000 ];
 [m,n] = size(cx);
 
 % data graph
 figure (1);
 plot(cx(:,1),cx(:,2),'k+');    hold on; %data
 plot(([0,0]),([-1,4]),'k-');   hold on; %X axis
 plot(([-1,4]),([0,0]),'k-');   hold on; %Y axis
 axis([-1,4,-1,4]);
 xlabel('Feature 1');
 ylabel('Feature 2');
 title('Original Data');
 
 %Covariance Matrix
 covX = cov(cx)
 %Covariance Matrix using the matrix definition
 meanX = mean(cx)
 
 cx1 = cx(:,1) - meanX(1);
 cx2 = cx(:,2) - meanX(2);
 
 Mcx = [cx1, cx2];
 covX = (transpose(Mcx)*(Mcx))/(m-1)   %definition of covariance
 
 %Covariance Matrix using alternative definition
 meanX = mean(cx);                     %mean of all elments of each row
 
 cx1 = cx(:,1);                        % substract mean of first row in cx
 cx2 = cx(:,2);                        % substract mean of second row in cx
 
 covX = ((transpose(cx)*(cx))/(m-1))-((transpose(meanX)*meanX)*(m/(m-1)))
 
 %Compute Eigenvalues and Eigenvector
 [W,L] = eig(covX)                     %W= Eigenvalues  L=Eigenvector
 
 % Eigenvector Graph
 figure(2);
 plot(cx(:,1),cx(:,2),'k+');                hold on;
 plot(([0,W(1,1)*4]),([0,W(1,2)*4]),'k-');  hold on;
 plot(([0,W(2,1)*4]),([0,W(2,2)*4]),'k-');
 axis([-4,4,-4,4]);
 xlabel('Feature 1');
 ylabel('Feature 2');
 title('Eigenvectors');
 
 %Transform Data
 cy = cx*transpose(W)
 
 %Graph Transformed Data
 figure(3);
 plot(cy(:,1),cy(:,2),'k+');      hold on;
 plot(([0,0]),([-1,5]),'k-');     hold on;
 plot(([-1,5]),([0,0]),'k-');
 axis([-1,5,-1,5]);
 xlabel('Feature 1');
 ylabel('Feature 2');
 title('Transformed Data');
 
 %Classification example
 meanY = mean(cy);
 
 % Graph of classification example
 figure(4)
 plot(([-5,5]),([meanY(2), meanY(2)]),'k:'); hold on;
 plot(([0,0]),([-5,5]),'k-');     hold on;
 plot(([-1,5]),([0,0]),'k-');     hold on;
 plot(cy(:,1),cy(:,2),'k+');      hold on;
 axis([-1,5,-1,5]);
 xlabel('Feature 1');
 ylabel('Feature 2');
 title('Classification Eample');
 legend('Mean',2);
 
 %Compression example
 cy(:,1)=zeros;
 xr = transpose (transpose(W)*transpose(cy));
 
 %Graph of compression example
 figure(5)
 plot(xr(:,1),xr(:,2),'k+');      hold on;
 plot(([0,0]),([-1,4]),'k-');     hold on;
 plot(([-1,4]),([0,0]),'k-');     
 axis([-1,4,-1,4]);
 xlabel('Feature 1');
 ylabel('Feature 2');
 title('Compression Eample');

 

 