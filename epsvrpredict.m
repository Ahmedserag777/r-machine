% Extreme Proximal SVR: Predict
%
% T=epsvrpredict(P,model)
%
% T=A predicted output Nx1
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
function T=epsvrpredict(P,model)
[N,n]=size(P);
PP1=[P ones(N,1)];
H=PP1*model.W1;
disp([min(H(:)) max(H(:))]);
hpi=pi/2;
switch model.a
    case 1%{'sin'}
        H=H*hpi;%i/p scaling
        H=sin(H);
    case 2%{'tanh'}
        H=H*hpi;
        H=tanh(H);
end
disp([min(H(:)) max(H(:))]);
T=H*model.w+model.r;
end