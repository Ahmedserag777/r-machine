% Backpropagation for 2 layer
%
% model=bptrain1(P,T,h,act,lr,mr,pr,maxIt,minSmE)
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
% lr=A learning rate for each layer(default=[0.001,0.001])
% mr=A momentum rate for each layer(default=[0.5,0.5])
% pr=A weight decaying ratio for each layer(default=[0,0])
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
function model=bptrain1(P,T,h,act,lr,mr,pr,maxIt,minSmE)
if nargin<9,
   minSmE=1e-8;
end
if nargin<8,
   maxIt=100;
end
if nargin<7,
   pr=[0,0];
end
if nargin<6,
   mr=[0.5,0.5];
end
if nargin<5,
   lr=[0.001,0.001];
end
if nargin<4,
   act=[2,2];
end
if nargin<3,
   h=5;
end
[N,n]=size(P);
[empty,m]=size(T);
L0=[P ones(N,1)];% adding bias input
model.W1=0.5*randn(n+1,h);
model.W2=0.5*randn(h+1,m);% generate random weights in the second layer -> (h+1)xm
%---Start
duW1=0*model.W1;
duW2=0*model.W2;
%---Calculate forward propagation
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
model.perf=zeros(maxIt,1);
it=1;% Set iteration=1
%---Calculate the results and display them
model.perf(it)=smE1;
disp([it model.perf(it)]);
while (it<=maxIt)&&(smE1>minSmE),
    %calculate the delta sum
      dW1=zeros(n+1,h);
      for ii=1:(n+1),
       for jj=1:h,
        for kk=1:N,
            dW1(ii,jj)=dW1(ii,jj)+dE1(jj,kk)*Q1(kk,jj)*L0(kk,ii);
        end
       end
      end
      dW1=-lr(1)*dW1;
      model.W1=model.W1+dW1+mr(1)*duW1-pr(1)*model.W1;
      duW1=dW1;
      dW2=zeros(h+1,m);
      for ii=1:(h+1),
       for jj=1:m,
        for kk=1:N,
            dW2(ii,jj)=dW2(ii,jj)+dE(kk,jj)*Q2(kk,jj)*L1(kk,ii);
        end
       end
      end    
      dW2=-lr(2)*dW2;
      model.W2=model.W2+dW2+mr(2)*duW2-pr(2)*model.W2;
      duW2=dW2;
%---Calculate forward propagation
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
%---Caluate results and display them    
      model.perf(it)=smE1;
      disp([it model.perf(it)]);   
      it=it+1;      
end
end