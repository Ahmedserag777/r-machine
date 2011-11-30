% Linear Proximal SVR train
%
% model=lpsvrtrain(P,T,v)
%
% model=A trained model
% P=Input patterns Nxm
% T=An output pattern (target) Nx1
% v=Regularized constant (default=1)
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
function model=lpsvrtrain(P,T,v)
if nargin<3,
   v=1;
end
[N,n]=size(P);
E=[P,ones(N,1)];
model.w=((E'*E+speye(n+1)/v)\E')*T;
model.r=model.w(n+1);
model.w=model.w(1:n);
end