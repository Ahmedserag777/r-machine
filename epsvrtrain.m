% Extreme Proximal SVR: Train
%
% model=epsvrtrain(P,T,h,a,v,br)
%
% model=A trained model
% P=Input patterns Nxn
% T=An output pattern (target) Nx1
% h=A number of hidden units
% a=An activation function 
%   1=tanh
%   2=sin
%   default(a=1)
% v=Regularized constant
% br=Bias constant
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
function model=epsvrtrain(P,T,h,a,v,br)
if nargin<6,
   br=1;
end
if nargin<5,
   v=1;
end
if nargin<4,
   a=1;%'tanh';
end
if nargin<3,
   h=10;
end
[N,n]=size(P);
PP1=[P ones(N,1)];
nn1=n+1;
W1=randn(nn1,h)+0.0000001;% Divided by zero prevention
% normalize each row of the weight sum
for hi=1:h,
    smW1=sum(abs(W1(1:n,hi)))+br*abs(W1(nn1,hi));
    W1(:,hi)=W1(:,hi)./smW1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
H=PP1*W1;
disp([min(H(:)) max(H(:))]);
hpi=pi/2;
switch a
    case 1%{'sin'}
        H=H*hpi;%i/p scaling
        H=sin(H);
    case 2%{'tanh'}
        H=H*hpi;
        H=tanh(H);
end
disp([min(H(:)) max(H(:))]);
Eh=[H,ones(size(H,1),1)];
Eh2=(Eh'*Eh);
Iv=eye(size(Eh2,1))/v;
W2=inv(Iv+Eh2)*Eh'*T;
model.W1=W1;
model.a=a;
model.w=W2(1:h);
model.r=W2(h+1);
end