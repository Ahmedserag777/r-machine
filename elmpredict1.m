% ELM Predict
%
% T=elmpredict1(P,model)
%
% T=Predicted outputs Nxm
% P=Input patterns Nxn
% model=A trained model
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
function T=elmpredict1(P,model)
[N,n]=size(P);
PP1=[P ones(N,1)];
H=PP1*model.W1;
disp([min(H(:)) max(H(:))]);
hpi=pi/2;
switch model.act(1)
    case 1%{'tanh'}
         H=H*hpi;
         H=tanh(H);    
    case 2%{'sin'}
         H=H*hpi;%i/p scaling
         H=sin(H);
end
disp([min(H(:)) max(H(:))]);
switch model.act(2)
    case 1%{'tanh'}
         T=tanh(H*model.B);    
    case 2%{'sin'}
         T=sin(H*model.B);
    case 3%{'lin'}
         T=H*model.B;
end         
end