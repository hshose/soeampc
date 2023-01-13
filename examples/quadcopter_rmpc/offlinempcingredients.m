%continous-time formulation
clear all;
clc;

tbxmanager restorepath

addpath('/home/hose/software/casadi')
import casadi.*

writeout = ~( getenv("WRITEOUT") == "");
if ~writeout
    disp("")
    disp("WARN: mpc parameters will NOT be written to file")
    disp("if you want to export, set environemt variable WRITEOUT")
end

%% grid over statespace
nx=10;
nu=3; 

g=9.81;
phivert = [ 0, tan(pi/4)^2;
            tan(pi/4)^2, 0;
            tan(pi/4)^2,tan(pi/4)^2;
            0,0];


% d0 = 10;  % Hu parameter
d0 = 80;    % berkeley parameter

d1 = 8;     % Hu and berkeley
% n0 = 10;  % Hu parameter
n0 = 40;    % berkeley parameter
kT=0.91;
m = 1.3;    % berkeley parameter
abl = [-d1, 1, 0, 0;
        -d0, 0, 0, 0;
        0, 0, -d1, 1;
        0, 0, -d0, 0];

%% setup variables for lmi optimization
con=[];
% rho_c=0.192;
% rho_c=0.0802;
rho_c=0.192*5;
Y = sdpvar(nu,nx);
X = sdpvar(nx);
% mpc stage cost
Q = diag([20,20,20, 1, 1, 1, 1e-2*ones(1,4)]);
R = diag([8,8,0.8]);
e=0;

%% iterate over grid and add lmi constraints
for i = 1:4
    abl = [-d1, 1, 0, 0;
            -d0, 0, 0, 0;
            0, 0, -d1, 1;
            0, 0, -d0, 0];
    A = [zeros(3,3), eye(3), zeros(3,4);
        zeros(1,6), g*(1+phivert(i,1)), 0,0, 0;
        zeros(1,6), 0, 0, g*(1+phivert(i,2)),0;
        zeros(1,10);
        zeros(4,6), abl ];
    B = [zeros(3,3);
        zeros(2,3);
        0, 0, kT/m;
        0,0,0;
        n0,0,0;
        0,0,0;
        0,n0,0];
    
    ineq=[(A*X+B*Y)+(A*X+B*Y)'+2*rho_c*X,   (sqrtm(Q+e*eye(nx))*X)',   (sqrtm(R)*Y)'; ...
            sqrtm(Q+e*eye(nx))*X,             -eye(nx),                   zeros(nx,nu);...
            sqrtm(R)*Y,                     zeros(nu,nx),                 -eye(nu)];
    con=[con;ineq<=0];
end
con=[con; X>=0];
    
%%
disp('Starting optimization');
ops = sdpsettings('solver','mosek','verbose',1,'debug',1)
optimize(con,-log(det(X)), ops) 
% optimize(con,-trace(X), ops) 
Y=value(Y);
X=value(X);
disp("X")
disp(X)
disp("eig(X)")
disp(eig(X))
P=inv(X);
K=Y*P;

% A = full(Afunc(zeros(nx,1), zeros(nu,1)));
% B = full(Bfunc(zeros(nx,1), zeros(nu,1)));
[K_lqr,P_lqr] = lqr(A,B,Q+e*eye(nx),R);

% LQR for initializing MPC
% [K_lqr_init,P_lqr_init] = lqr(A,B,diag([1,1,0.1, 1, 1, 1e-3, 1, 1e-1, 1, 1e-1]),diag([1e-3, 1e-3,1e-3]));
% [K_lqr_init,P_lqr_init] = lqr(A,B,diag([1,1,0.1, 1, 1, 1e-3, 1, 1e-1, 1, 1e-1]),diag([1e-3, 1e-3,1e-3]));
[K_lqr_init,P_lqr_init] = lqr(A,B,Q,R*0.05);
if writeout
    writematrix(reshape(round(-K_lqr_init,6),1,[]), 'mpc_parameters/Kinit.txt');
end

disp("max(eig(P/P_lqr))")
disp(eigs(P/P_lqr,1))
disp("max(eig(P))")
disp(max(eig(P)))
disp("min(eig(P))")
disp(min(eig(P)))
% % disp("alpha")
% % disp(alpha)
% % save('cont_constant.mat','P_delta','K_delta'); 

% % x_max = 0.1*[deg2rad(10), deg2rad(10), deg2rad(360), deg2rad(360), deg2rad(360), 200*2*pi/60, 2000*2*pi/60];
% % x_min = -x_max;
% % u_max = 1*[1.6, 1.6];
% % u_min = -u_max;


%% Terminal Set
% constraints of form L*[x;u] <= l
% first construct Lx*x + Lu*u <= 1

Lx = zeros(11,nx);
Lu = zeros(11,nu);
l = ones(11,1);

pxmax = 0.145;
% phimax = pi/2;
phimax = 20*pi/180;
u12max = 45*pi/180;
u3min = -g/kT;
u3max = 18-g/kT;
Lx(1,1) = 1/pxmax;
Lx(2,7) = 1/phimax;
Lx(3,9) = 1/phimax;
Lx(4,7) = 1/-phimax;
Lx(5,9) = 1/-phimax;
Lu(5+1,1) = 1/u12max;
Lu(5+2,2) = 1/u12max;
Lu(5+3,3) = 1/u3max;
Lu(5+4,1) = 1/-u12max;
Lu(5+5,2) = 1/-u12max;
Lu(5+6,3) = 1/u3min;

L = [Lx, Lu];

% L = [eye(nx), zeros(nx,nu);
%     -eye(nx), zeros(nx,nu);
%     zeros(nu,nx),  eye(nu);
%     zeros(nu,nx), -eye(nu) ];
% l = [x_max'; -x_min'; u_max'; -u_min'];
Nz = length(l);

r = zeros(nx+nu,1);
C = zeros(Nz,1);
C_lqr = zeros(Nz,1);
rhs = zeros(Nz,1);
for k=1:Nz
    C(k) = norm(inv(sqrtm(P))*[eye(nx), K']*L(k,:)');
    C_lqr(k) = norm(inv(sqrtm(P_lqr))*[eye(nx), -K_lqr']*L(k,:)');
    rhs(k) = (l(k)-L(k,:)*r);
end

res = rhs./C;
res_lqr = rhs./C_lqr;

[alpha, idx] = min(res);
[alpha_lqr, idx_lqr] = min(res_lqr);

disp("alpha")
disp(alpha)

disp("active constraint number")
disp(idx)

% disp("maximum psi dot [deg/s]")
% disp(rad2deg(sqrt(norm(inv(sqrtm(P))*[eye(7), K']*L(3,:)')^2*alpha)))

% disp("maximum omega_R [rpm]")
% disp(sqrt(norm(inv(sqrtm(P))*[eye(7), K']*L(7,:)')^2*alpha)*60/2/pi)

if writeout
    writematrix(reshape(round(P,6),1,[]), 'mpc_parameters/P.txt');
    writematrix(reshape(round(Q,6),1,[]), 'mpc_parameters/Q.txt');
    writematrix(reshape(round(K,6),1,[]), 'mpc_parameters/K.txt');
    writematrix(reshape(round(R,6),1,[]), 'mpc_parameters/R.txt');
    writematrix(alpha,             'mpc_parameters/alpha.txt');
end


% Constraint Tighetning

con=[];
% c_u=1;
% rho_c=0.0802;
wbar=1;
Y = sdpvar(nu,nx);
X = sdpvar(nx);


uw = 5e-3*[  1,  1,  1;
             1,  1, -1;
             1, -1,  1;
             1, -1, -1;
            -1,  1,  1;
            -1,  1, -1;
            -1, -1,  1;
            -1, -1, -1];
dw = (B*uw')';

for i = 1:4
    A = [zeros(3,3), eye(3), zeros(3,4);
        zeros(1,6), g*(1+phivert(i,1)), 0,0, 0;
        zeros(1,6), 0, 0, g*(1+phivert(i,2)),0;
        zeros(1,10);
        zeros(4,6), abl ];
    B = [zeros(3,3);
        zeros(2,3);
        0, 0, kT;
        0,0,0;
        n0,0,0;
        0,0,0;
        0,n0,0];
    ineq=[(A*X+B*Y)+(A*X+B*Y)'+2*rho_c*X];
    con=[con;ineq<=0];
end
con=[con; X>=0];

for j=1:size(Lx,1)
    ineq=[1,Lx(j,:)*X+Lu(j,:)*Y;...
        (Lx(j,:)*X+Lu(j,:)*Y)',X];
    con=[con;ineq>=0];
end

for j=1:size(dw,1)
    ineq=[X, dw(j,:)';
        dw(j,:), wbar^2];
    con=[con;ineq>=0];
end



disp('Starting optimization');
optimize(con,-log(det(X)))
% 
Y=value(Y);
X=value(X);
Pdelta=X^-1
Kdelta=Y*X^-1

% dmax = sqrt(wbar/Pdelta(2,2))
wbarmin = dw(1,:)*Pdelta*dw(1,:)'

if writeout
    writematrix(round(wbarmin, 6),                'mpc_parameters/wbar.txt')
    writematrix(round(rho_c, 6),                  'mpc_parameters/rho_c.txt')
    writematrix(reshape(round(Pdelta,6),1,[]), 'mpc_parameters/Pdelta.txt')
    writematrix(reshape(round(Kdelta,6),1,[]), 'mpc_parameters/Kdelta.txt')
end

cj = [];
for i = 1:size(L,1)
    % norm(inv(sqrtm(P))*[eye(nx), K']*L(k,:)')^2;
    cj = [cj; norm(inv(sqrtm(Pdelta))*[eye(nx), Kdelta']*L(i,:)')];
end

c_max = max(cj);

if writeout
    writematrix(reshape(round(cj,6),1,[]), 'mpc_parameters/Ls.txt')
    writematrix(reshape(round(Lx,6),1,[]), 'mpc_parameters/Lx.txt')
    writematrix(reshape(round(Lu,6),1,[]), 'mpc_parameters/Lu.txt')
end

alpha_s = norm(sqrtm(P)*inv(sqrtm(Pdelta)));
Tf = 1
alpha - alpha_s*(1-exp(-rho_c*Tf))/rho_c*wbarmin

if writeout
    writematrix(Tf,     'mpc_parameters/Tf.txt')
    writematrix(alpha_s,'mpc_parameters/alpha_s.txt');
end