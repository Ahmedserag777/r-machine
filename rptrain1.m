% Resilient propagation for 2-layer networks
%
% model=rptrain1(P,T,h,act,pr,srU,srD,lrU,lrD,d0,maxIt,minSmE)
%
% model=A trained model
% P=Input patterns Nxn
% T=Output patterns (target) Nxm
% h=A number of hidden units
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
function model=rptrain1(P,T,h,act,pr,srU,srD,lrU,lrD,d0,maxIt,minSmE)
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
   lrD=[1e-24,1e-24];
end   
if nargin<8,
   lrU=[1e24,1e24];
end   
if nargin<7,
   srD=[0.5,0.5];
end
if nargin<6,
   srU=[1.2,1.2];
end
if nargin<5,
   pr=[0,0];
end
if nargin<4,
   act=[2,3];
end
if nargin<3,
   h=5;
end
[N,n]=size(P);
[empty,m]=size(T);
L0=[P ones(N,1)];
model.W1=0.1*randn(n+1,h);
model.W2=0.1*randn(h+1,m);
%---Start
dD1=0*model.W1;
dD2=0*model.W2;
duD1=dD1+d0;
duD2=dD2+d0;
duW1=0*model.W1;
duW2=0*model.W2;
%---Forward propagation and back propagation error
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
dE=(O2-T);
dE1=model.W2*dE';
smE1=0.5*sum(sum(dE.^2,1),2);
%---End forward propagation and back propagation error
model.perf=zeros(maxIt,1);
it=1;% Set iteration=1
%---Keep and display result
model.perf(it)=smE1;
disp([it model.perf(it)]);
while (it<=maxIt)&&(smE1>minSmE),
%---Delta sum of weights 1
      dW1=zeros(n+1,h);
      for ii=1:(n+1),
       for jj=1:h,
        for kk=1:N,
            dW1(ii,jj)=dW1(ii,jj)+dE1(jj,kk)*Q1(kk,jj)*L0(kk,ii);
        end
       end
      end
%---End delta sum of weights 1      
%---Update weights 1
      for ii=1:(n+1),
       for jj=1:h,
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
%---End update weights 1
%---Delta sum of weights 2 
      dW2=zeros(h+1,m);
      for ii=1:(h+1),
       for jj=1:m,
        for kk=1:N,
            dW2(ii,jj)=dW2(ii,jj)+dE(kk,jj)*Q2(kk,jj)*L1(kk,ii);
        end
       end
      end    
%---End delta sum of weights 2
%---Update weigths 2
      for ii=1:(h+1),
       for jj=1:m,
           dWW=duW2(ii,jj)*dW2(ii,jj);
           if dWW>0,
              dD2(ii,jj)=srU(2)*duD2(ii,jj);
              dD2(ii,jj)=min([dD2(ii,jj),lrU(2)]);
           elseif dWW<0,
                  dD2(ii,jj)=srD(2)*duD2(ii,jj);
                  dD2(ii,jj)=max([dD2(ii,jj),lrD(1)]);
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
%---End update weights 2    
%---Forward propagation and back propagation error
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
      dE=(O2-T);% Output differences
      dE1=model.W2*dE';% Error back propagation via layer 2
      smE1=0.5*sum(sum(dE.^2,1),2);% Sum of square error
%---End forward fropagation with back propagation error
%---Keep and display result    
      model.perf(it)=smE1;
      disp([it model.perf(it)]);   
      it=it+1;
end    
end