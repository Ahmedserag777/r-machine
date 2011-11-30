% Resilient propagation for 6-layer networks
%
% model=rptrain5(P,T,h,act,pr,srU,srD,lrU,lrD,d0,maxIt,minSmE)
%
% model=A trained model
% P=Input patterns Nxn
% T=Output patterns (target) Nxm
% h=A number of hidden units in vector. 
%   Example, if h=[5,6] then 1st-layer=5 and 2nd-layer=6
% act=An activation function in vector
%     1=tanh
%     2=sin
%     3=linear
%     Example, if act=[2,3] then 1st-layer=tanh and last layer=linear
% pr=A weight decaying ratio for each layer(default=[0,0])
% srU=An up compensation ratio for each layer(default=[1.2,1.2])
% srD=A down compensation ratio for each layer(default=[0.5,0.5])
% lrU=An up limit for each layer(default=[1e24,1e24])
% lrD=A down limit for each layer(default=[1e-24,1e-24])
% d0=delta start (default=0.01)
% maxIt=Maximum iteration (default=100)
% minSmE=Minimum sum of square error (default=1e-8)
%
% Copyright by Pat Taweewat
% in R-Machine
%   This program is free software: you can redistribute it and/or modify
%   it under the terms of the GNU General Public License as published by
%   the Free Software Foundation, either version 3 of the License, or
%   (at your option) any later version.
%
%   This program is distributed in the hope that it will be useful,
%   but WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%   GNU General Public License for more details.
%
%   You should have received a copy of the GNU General Public License
%   along with this program.  If not, see <http://www.gnu.org/licenses/>.
%
function model=rptrain5(P,T,h,act,pr,srU,srD,lrU,lrD,d0,maxIt,minSmE)
if nargin<12,
   minSmE=1e-8;
end
if nargin<11,
   maxIt=100;
end
if nargin<10,
   d0=0.01;
end
if nargin<9,
   lrD=[1e-24,1e-24,1e-24,1e-24,1e-24,1e-24];
end   
if nargin<8,
   lrU=[1e24,1e24,1e24,1e24,1e24,1e24];
end   
if nargin<7,
   srD=[0.5,0.5,0.5,0.5,0.5,0.5];
end
if nargin<6,
   srU=[1.2,1.2,1.2,1.2,1.2,1.2];
end
if nargin<5,
   pr=[0,0,0,0,0,0];
end
if nargin<4,
   act=[2,2,2,2,2,3];
end
if nargin<3,
   h=[10,10,10,10,10];
end
[N,n]=size(P);
[empty,m]=size(T);
L0=[P ones(N,1)];
model.W1=0.1*randn(n+1,h(1));
model.W2=0.1*randn(h(1)+1,h(2));
model.W3=0.1*randn(h(2)+1,h(3));
model.W4=0.1*randn(h(3)+1,h(4));
model.W5=0.1*randn(h(4)+1,h(5));
model.W6=0.1*randn(h(5)+1,m);
%---Start
dD1=0*model.W1;
dD2=0*model.W2;
dD3=0*model.W3;
dD4=0*model.W4;
dD5=0*model.W5;
dD6=0*model.W6;
duD1=dD1+d0;
duD2=dD2+d0;
duD3=dD3+d0;
duD4=dD4+d0;
duD5=dD5+d0;
duD6=dD6+d0;
duW1=0*model.W1;
duW2=0*model.W2;
duW3=0*model.W3;
duW4=0*model.W4;
duW5=0*model.W5;
duW6=0*model.W6;
%---Calculate forward propagation and back propagation error
S1=L0*model.W1;
switch act(1)
    case 1%tanh
         O1=tanh(S1);
         Q1=1-(O1.*O1);    
    case 2%sin
         O1=sin(S1);
         Q1=cos(S1);
    case 3%linear
         O1=S1;
         Q1=ones(size(S1));
end         
L1=[O1 ones(N,1)];
S2=L1*model.W2;
switch act(2)
    case 1%tanh
         O2=tanh(S2);
         Q2=1-(O2.*O2);    
    case 2%sin
         O2=sin(S2);
         Q2=cos(S2);
    case 3%linear
         O2=S2;
         Q2=ones(size(S2));         
end
L2=[O2 ones(N,1)];
S3=L2*model.W3;
switch act(3)
    case 1%tanh
         O3=tanh(S3);
         Q3=1-(O3.*O3);    
    case 2%sin
         O3=sin(S3);
         Q3=cos(S3);
    case 3%linear
         O3=S3;
         Q3=ones(size(S3));         
end
L3=[O3 ones(N,1)];
S4=L3*model.W4;
switch act(4)
    case 1%tanh
         O4=tanh(S4);
         Q4=1-(O4.*O4);    
    case 2%sin
         O4=sin(S4);
         Q4=cos(S4);
    case 3%linear
         O4=S4;
         Q4=ones(size(S4));         
end
L4=[O4 ones(N,1)];
S5=L4*model.W5;
switch act(5)
    case 1%tanh
         O5=tanh(S5);
         Q5=1-(O5.*O5);    
    case 2%sin
         O5=sin(S5);
         Q5=cos(S5);
    case 3%linear
         O5=S5;
         Q5=ones(size(S5));         
end
L5=[O5 ones(N,1)];
S6=L5*model.W6;
switch act(6)
    case 1%tanh
         O6=tanh(S6);
         Q6=1-(O6.*O6);    
    case 2%sin
         O6=sin(S6);
         Q6=cos(S6);
    case 3%linear
         O6=S6;
         Q6=ones(size(S6));         
end
dE=(O6-T);
dE5=model.W6*dE';
dE4=model.W5*dE5(1:h(5),:);
dE3=model.W4*dE4(1:h(4),:);
dE2=model.W3*dE3(1:h(3),:);
dE1=model.W2*dE2(1:h(2),:);
smE1=0.5*sum(sum(dE.^2,1),2);
%---End calculation forward propagation and back propagation error
model.perf=zeros(maxIt,1);
it=1;% Set iteration=1
%---Calculate the results and display them
model.perf(it)=smE1;
disp([it model.perf(it)]);
while (it<=maxIt)&&(smE1>minSmE),
    %calculate the delta sum
    dW1=zeros(n+1,h(1));
    for ii=1:(n+1),
     for jj=1:h(1),
      for kk=1:N,
          dW1(ii,jj)=dW1(ii,jj)+dE1(jj,kk)*Q1(kk,jj)*L0(kk,ii);
      end
     end
    end
    for ii=1:(n+1),
     for jj=1:h(1),
         dWW=duW1(ii,jj)*dW1(ii,jj);
         if dWW>0,
            dD1(ii,jj)=srU(1)*duD1(ii,jj);
            dD1(ii,jj)=min([dD1(ii,jj),lrU(1)]);             
         elseif dWW<0,
                dD1(ii,jj)=srD(1)*duD1(ii,jj);
                dD1(ii,jj)=max([dD1(ii,jj),lrD(1)]);                  
         elseif dWW==0,
                dD1(ii,jj)=duD1(ii,jj);
         end
         if dW1(ii,jj)>0,
            model.W1(ii,jj)=model.W1(ii,jj)-dD1(ii,jj);
         elseif dW1(ii,jj)<0,
                model.W1(ii,jj)=model.W1(ii,jj)+dD1(ii,jj);
         end
     end
    end
    model.W1=model.W1-pr(1)*model.W1;
    duW1=dW1;
    duD1=dD1;
    
    dW2=zeros(h(1)+1,h(2));
    for ii=1:(h(1)+1),
     for jj=1:h(2),
      for kk=1:N,
          dW2(ii,jj)=dW2(ii,jj)+dE2(jj,kk)*Q2(kk,jj)*L1(kk,ii);
      end
     end
    end    

    for ii=1:(h(1)+1),
     for jj=1:h(2),
         dWW=duW2(ii,jj)*dW2(ii,jj);
         if dWW>0,
            dD2(ii,jj)=srU(2)*duD2(ii,jj);
            dD2(ii,jj)=min([dD2(ii,jj),lrU(2)]);             
         elseif dWW<0,
                dD2(ii,jj)=srD(2)*duD2(ii,jj);
                dD2(ii,jj)=max([dD2(ii,jj),lrD(2)]);                  
         elseif dWW==0,
                dD2(ii,jj)=duD2(ii,jj);
         end
         if dW2(ii,jj)>0,
            model.W2(ii,jj)=model.W2(ii,jj)-dD2(ii,jj);
         elseif dW2(ii,jj)<0,
                model.W2(ii,jj)=model.W2(ii,jj)+dD2(ii,jj);
         end
     end
    end
    model.W2=model.W2-pr(2)*model.W2;
    duW2=dW2;
    duD2=dD2;

    dW3=zeros(h(2)+1,h(3));
    for ii=1:(h(2)+1),
     for jj=1:h(3),
      for kk=1:N,
          dW3(ii,jj)=dW3(ii,jj)+dE3(jj,kk)*Q3(kk,jj)*L2(kk,ii);
      end
     end
    end        
    for ii=1:(h(2)+1),
     for jj=1:h(3),
         dWW=duW3(ii,jj)*dW3(ii,jj);
         if dWW>0,
            dD3(ii,jj)=srU(3)*duD3(ii,jj);
            dD3(ii,jj)=min([dD3(ii,jj),lrU(3)]);             
         elseif dWW<0,
                dD3(ii,jj)=srD(3)*duD3(ii,jj);
                dD3(ii,jj)=max([dD3(ii,jj),lrD(3)]);                  
         elseif dWW==0,
                dD3(ii,jj)=duD3(ii,jj);
         end
         if dW3(ii,jj)>0,
            model.W3(ii,jj)=model.W3(ii,jj)-dD3(ii,jj);
         elseif dW3(ii,jj)<0,
                model.W3(ii,jj)=model.W3(ii,jj)+dD3(ii,jj);
         end
     end
    end
    model.W3=model.W3-pr(3)*model.W3;
    duW3=dW3;
    duD3=dD3;
    
    dW4=zeros(h(3)+1,h(4));
    for ii=1:(h(3)+1),
     for jj=1:m,
      for kk=1:N,
          dW4(ii,jj)=dW4(ii,jj)+dE4(jj,kk)*Q4(kk,jj)*L3(kk,ii);
      end
     end
    end        
    for ii=1:(h(3)+1),
     for jj=1:h(4),
         dWW=duW4(ii,jj)*dW4(ii,jj);
         if dWW>0,
            dD4(ii,jj)=srU(4)*duD4(ii,jj);
            dD4(ii,jj)=min([dD4(ii,jj),lrU(4)]);             
         elseif dWW<0,
                dD4(ii,jj)=srD(4)*duD4(ii,jj);
                dD4(ii,jj)=max([dD4(ii,jj),lrD(4)]);                  
         elseif dWW==0,
                dD4(ii,jj)=duD4(ii,jj);
         end
         if dW4(ii,jj)>0,
            model.W4(ii,jj)=model.W4(ii,jj)-dD4(ii,jj);
         elseif dW4(ii,jj)<0,
                model.W4(ii,jj)=model.W4(ii,jj)+dD4(ii,jj);
         end
     end
    end
    model.W4=model.W4-pr(4)*model.W4;
    duW4=dW4;
    duD4=dD4;

    dW5=zeros(h(4)+1,h(5));
    for ii=1:(h(4)+1),
     for jj=1:h(5),
      for kk=1:N,
          dW5(ii,jj)=dW5(ii,jj)+dE5(jj,kk)*Q5(kk,jj)*L4(kk,ii);
      end
     end
    end        
    for ii=1:(h(4)+1),
     for jj=1:h(5),
         dWW=duW5(ii,jj)*dW5(ii,jj);
         if dWW>0,
            dD5(ii,jj)=srU(5)*duD5(ii,jj);
            dD5(ii,jj)=min([dD5(ii,jj),lrU(5)]);             
         elseif dWW<0,
                dD5(ii,jj)=srD(5)*duD5(ii,jj);
                dD5(ii,jj)=max([dD5(ii,jj),lrD(5)]);                  
         elseif dWW==0,
                dD5(ii,jj)=duD5(ii,jj);
         end
         if dW5(ii,jj)>0,
            model.W5(ii,jj)=model.W5(ii,jj)-dD5(ii,jj);
         elseif dW5(ii,jj)<0,
                model.W5(ii,jj)=model.W5(ii,jj)+dD5(ii,jj);
         end
     end
    end
    model.W5=model.W5-pr(5)*model.W5;
    duW5=dW5;
    duD5=dD5;            

    dW6=zeros(h(5)+1,m);
    for ii=1:(h(5)+1),
     for jj=1:m,
      for kk=1:N,
          dW6(ii,jj)=dW6(ii,jj)+dE(kk,jj)*Q6(kk,jj)*L5(kk,ii);
      end
     end
    end        
    for ii=1:(h(5)+1),
     for jj=1:m,
         dWW=duW6(ii,jj)*dW6(ii,jj);
         if dWW>0,
            dD6(ii,jj)=srU(6)*duD6(ii,jj);
            dD6(ii,jj)=min([dD6(ii,jj),lrU(6)]);             
         elseif dWW<0,
                dD6(ii,jj)=srD(6)*duD6(ii,jj);
                dD6(ii,jj)=max([dD6(ii,jj),lrD(6)]);                  
         elseif dWW==0,
                dD6(ii,jj)=duD6(ii,jj);
         end
         if dW6(ii,jj)>0,
            model.W6(ii,jj)=model.W6(ii,jj)-dD6(ii,jj);
         elseif dW6(ii,jj)<0,
                model.W6(ii,jj)=model.W6(ii,jj)+dD6(ii,jj);
         end
     end
    end
    model.W6=model.W6-pr(6)*model.W6;
    duW6=dW6;
    duD6=dD6;
%---Calculate forward propagation and back propagation error
S1=L0*model.W1;
switch act(1)
    case 1%tanh
         O1=tanh(S1);
         Q1=1-(O1.*O1);    
    case 2%sin
         O1=sin(S1);
         Q1=cos(S1);
    case 3%linear
         O1=S1;
         Q1=ones(size(S1));
end         
L1=[O1 ones(N,1)];
S2=L1*model.W2;
switch act(2)
    case 1%tanh
         O2=tanh(S2);
         Q2=1-(O2.*O2);    
    case 2%sin
         O2=sin(S2);
         Q2=cos(S2);
    case 3%linear
         O2=S2;
         Q2=ones(size(S2));         
end
L2=[O2 ones(N,1)];
S3=L2*model.W3;
switch act(3)
    case 1%tanh
         O3=tanh(S3);
         Q3=1-(O3.*O3);    
    case 2%sin
         O3=sin(S3);
         Q3=cos(S3);
    case 3%linear
         O3=S3;
         Q3=ones(size(S3));         
end
L3=[O3 ones(N,1)];
S4=L3*model.W4;
switch act(4)
    case 1%tanh
         O4=tanh(S4);
         Q4=1-(O4.*O4);    
    case 2%sin
         O4=sin(S4);
         Q4=cos(S4);
    case 3%linear
         O4=S4;
         Q4=ones(size(S4));         
end
L4=[O4 ones(N,1)];
S5=L4*model.W5;
switch act(5)
    case 1%tanh
         O5=tanh(S5);
         Q5=1-(O5.*O5);    
    case 2%sin
         O5=sin(S5);
         Q5=cos(S5);
    case 3%linear
         O5=S5;
         Q5=ones(size(S5));         
end
L5=[O5 ones(N,1)];
S6=L5*model.W6;
switch act(6)
    case 1%tanh
         O6=tanh(S6);
         Q6=1-(O6.*O6);    
    case 2%sin
         O6=sin(S6);
         Q6=cos(S6);
    case 3%linear
         O6=S6;
         Q6=ones(size(S6));         
end
dE=(O6-T);
dE5=model.W6*dE';
dE4=model.W5*dE5(1:h(5),:);
dE3=model.W4*dE4(1:h(4),:);
dE2=model.W3*dE3(1:h(3),:);
dE1=model.W2*dE2(1:h(2),:);
smE1=0.5*sum(sum(dE.^2,1),2);
%---End calculation forward propagation and back propagation error    
%---Caluate results and display them    
      model.perf(it)=smE1;
      disp([it model.perf(it)]);   
      it=it+1;          
end
end