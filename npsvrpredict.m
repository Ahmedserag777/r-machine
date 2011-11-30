% Non-linear Proximal SVR predict
%
% T=npsvrpredict(P,model)
%
% T=A predicted output Nx1
% P=Input patterns Nxm
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
function T=npsvrpredict(P,model)
N=size(model.P,1);
NN=size(P,1);
model.P=model.P';
P=P';
K=zeros(NN,N);
for ii=1:NN,
 for jj=1:N,
     %K(ii,jj)=model.P(:,jj)'*P(:,ii);% Linear
     %K(ii,jj)=exp((-norm(P(:,ii)-model.P(:,jj))^2)/mo2);% RBF
     switch model.kr(1)
         case 1%linear
              K(ii,jj)=model.P(:,jj)'*P(:,ii);
         case 2%RBF
              K(ii,jj)=exp((-norm(P(:,ii)-model.P(:,jj))^2)/0.5);
         case 3%Polynomial
              K(ii,jj)=(model.P(:,jj)'*P(:,ii)+1)^2;              
         case 4%Perceptron
              K(ii,jj)=tanh(0.5*model.P(:,jj)'*P(:,ii)+0.5);                           
         case 5%RBF with option
              K(ii,jj)=exp((-norm(P(:,ii)-model.P(:,jj))^2)/model.kr(2));
         case 6%Polynomial with option
              K(ii,jj)=(model.P(:,jj)'*P(:,ii)+1)^model.kr(2);  
         case 7%Perceptron with option
              K(ii,jj)=tanh(model.kr(2)*model.P(:,jj)'*P(:,ii)+model.kr(3));
     end        
 end
end
KO=zeros(N,N);
for ii=1:N,
 for jj=1:N,
     %KO(ii,jj)=model.P(:,jj)'*model.P(:,ii);% Linear
     %KO(ii,jj)=exp((-norm(model.P(:,ii)-model.P(:,jj))^2)/mo2);% RBF
     switch model.kr(1)
         case 1%linear
              KO(ii,jj)=model.P(:,jj)'*model.P(:,ii);
         case 2%RBF
              KO(ii,jj)=exp((-norm(model.P(:,ii)-model.P(:,jj))^2)/0.5);
         case 3%Polynomial
              KO(ii,jj)=(model.P(:,jj)'*model.P(:,ii)+1)^2;              
         case 4%Perceptron
              KO(ii,jj)=tanh(0.5*model.P(:,jj)'*model.P(:,ii)+0.5);                           
         case 5%RBF with option
              KO(ii,jj)=exp((-norm(model.P(:,ii)-model.P(:,jj))^2)/model.kr(2));
         case 6%Polynomial with option
              KO(ii,jj)=(model.P(:,jj)'*model.P(:,ii)+1)^model.kr(2);  
         case 7%Perceptron with option
              KO(ii,jj)=tanh(model.kr(2)*model.P(:,jj)'*model.P(:,ii)+model.kr(3));
     end             
 end
end
T=(K*KO'+ones(NN,N))*model.b;
end