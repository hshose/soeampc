%continous-time formulation
clear all;
clc;

addpath('/home/hose/software/casadi')
import casadi.*

%% grid over statespace
nx=10;
nu=3; 

% alpha_f = 9.2e-5
% eps     = 2.2e-3
% ue      = 0.7853
% xe1     = 0.2632
% xe2     = 0.6519

g=9.8

phivert = [ 0, tan(pi/4)^2;
            tan(pi/4)^2, 0;
            tan(pi/4)^2,tan(pi/4)^2;
            0,0]

% disp("Grid size x")
% disp(N_xgrid)
% disp("Grid size u")
% disp(N_ugrid)


%% setup variables for lmi optimization
con=[];
c_u=1;
rho_c=0.0802;
wbar=1;
Y = sdpvar(nu,nx);
X = sdpvar(nx);
e = 0.0;
% mpc stage cost
Q = diag([1,1,1,1e-2*ones(1,7)]);
R = 1e-3*eye(3);
e=0;

Lx = [ 1/x_max(1),  0;
    1/x_min(1),  0;
    0 1/x_max(2);
    0 1/x_min(2);
    zeros(2,2)];
Lu = [ zeros(4,1);
    1/u_max(1);
    1/u_min(1)];

dw = [0, 0.001;
      0, -0.001;];

for i = 1:N_xgrid
    for j = 1:N_ugrid
        px = [X1(i),X2(i)];
        pu = U(j);
        A = full(Afunc(px, pu));
        B = full(Bfunc(px, pu));

        ineq=[(A*X+B*Y)+(A*X+B*Y)'+2*rho_c*X];
        con=[con;ineq<=0];
    end
end
con=[con; X>=0];

for j=1:size(Lx,1)
    ineq=[1,Lx(j,:)*X+Lu(j,:)*Y;...
        (Lx(j,:)*X+Lu(j,:)*Y)',X];
    con=[con;ineq>=0];
end

% for j=1:size(dw,1)
%     ineq=[X, dw(j,:)';
%         dw(j,:), wbar^2]
%     con=[con;ineq>=0];
% end

disp('Starting optimization');
optimize(con,-log(det(X)))
% 
Y=value(Y);
X=value(X);
Pdelta=X^-1
Kdelta=Y*X^-1

dmax = sqrt(wbar/Pdelta(2,2))
wbarmin = dw(1,:)*Pdelta*dw(1,:)'


writematrix(wbarmin, 'parameter/wbar.txt')
writematrix(rho_c, 'parameter/rho_c.txt')
writematrix(reshape(Pdelta.',1,[]), 'parameter/Pdelta.txt')
writematrix(reshape(Kdelta.',1,[]), 'parameter/Kdelta.txt')


cj = [];
for i = 1:size(Lx,1)
    cj = [cj; norm(Pdelta^-(1/2)*(Lx(i,:)'+Kdelta'*Lu(i,:)'),2)]
end

c_max = max(cj)

writematrix(reshape(cj.',1,[]), 'parameter/Ls.txt')