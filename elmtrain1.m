% ELM Train
%
% model=elmtrain1(P,T,h,act,br)
%
% model=A trained model
% 
% P=Input patterns. Nxn
% T=Output patterns (targets). Nxm
% h=A number of hidden-unit neurons.
% act=An activation function in vector. default=tanh
% br=A bias ratio in sum(abs(W1))+br*abs(W1(n+1));
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
function model=elmtrain1(P,T,h,act,br)
if nargin<5,
   br=1;
end
if nargin<4,
   act=[2,3];
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
switch act(1)
    case 1%{'tanh'}
         H=H*hpi;
         H=tanh(H);    
    case 2%{'sin'}
         H=H*hpi;%i/p scaling
         H=sin(H);
end
disp([min(H(:)) max(H(:))]);
%model.B=inv(H'*H)*H'*T;
switch act(2)
    case 1%{'tanh'}
         model.B=pinv(H)*atanh(T*0.999999);    
    case 2%{'sin'}
         model.B=pinv(H)*asin(T);
    case 3%{'lin'}
         model.B=pinv(H)*T;
end 
model.W1=W1;
model.act=act;
end