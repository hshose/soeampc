
%continous-time formulation

tbxmanager restorepath

clear all;
clc;

addpath('/home/hose/software/casadi')
import casadi.*


%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%
%% Terminal set %%
%%%%%%%%%%%%%%%%%%

%% grid over statespace
nx=2;nu=1; 

ue      = 0.7853
xe1     = 0.2632
xe2     = 0.6519

u_min = [0-ue];
u_max = [2-ue];
x_min = 0.40*[-0.2,-0.2];
x_max = -x_min;

x_step = (x_max-x_min)/30
u_step = (u_max-u_min)/30

[X1, X2] = ndgrid(x_min:x_step:x_max);
[U] = ndgrid(u_min:u_step:u_max);
N_xgrid = numel(X1);
N_ugrid = numel(U);

disp("Grid size x")
disp(N_xgrid)
disp("Grid size u")
disp(N_ugrid)

%% system dynamics in casadi
x = SX.sym('x',nx);
u = SX.sym('u',nu);
f = Function('f', {x,u}, {stirtank_dynamics(x(1), x(2), u) } );
Afunc = Function('A', {x,u}, {jacobian(f(x,u), x)});
Bfunc = Function('B', {x,u}, {jacobian(f(x,u), u)});

%% setup variables for lmi optimization
con=[];
rho_c=0.05;
Y = sdpvar(nu,nx);
X = sdpvar(nx);
e = 0.;
% mpc stage cost
Q = eye(nx);
R = 1;

%% iterate over grid and add lmi constraints
for i = 1:N_xgrid
    for j = 1:N_ugrid
        px = [X1(i),X2(i)];
        pu = U(j);
        A = full(Afunc(px, pu));
        B = full(Bfunc(px, pu));
        ineq=[(A*X+B*Y)+(A*X+B*Y)'+2*rho_c*X,   (sqrtm(Q+e*eye(nx))*X)',   (sqrtm(R)*Y)'; ...
                sqrtm(Q+e*eye(nx))*X,             -eye(nx),                   zeros(nx,nu);...
                sqrtm(R)*Y,                     zeros(nu,nx),                 -eye(nu)];
        con=[con;ineq<=0];
    end
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

A = full(Afunc(zeros(nx,1), zeros(nu,1)));
B = full(Bfunc(zeros(nx,1), zeros(nu,1)));
[K_lqr,P_lqr] = lqr(A,B,Q+e*eye(nx),R);

disp("max(eig(P/P_lqr))")
disp(eigs(P/P_lqr,1))
disp("max(eig(P))")
disp(max(eig(P)))
disp("min(eig(P))")
disp(min(eig(P)))
% disp("alpha")
% disp(alpha)
% save('cont_constant.mat','P_delta','K_delta'); 

% x_max = 0.1*[deg2rad(10), deg2rad(10), deg2rad(360), deg2rad(360), deg2rad(360), 200*2*pi/60, 2000*2*pi/60];
% x_min = -x_max;
% u_max = 1*[1.6, 1.6];
% u_min = -u_max;


%% Terminal Set
% constraints of form L*[x;u] <= l
L = [eye(nx), zeros(nx,nu);
    -eye(nx), zeros(nx,nu);
    zeros(nu,nx),  eye(nu);
    zeros(nu,nx), -eye(nu) ];
l = [x_max'; -x_min'; u_max'; -u_min'];
Nz = length(l);

r = zeros(nx+nu,1);
C = zeros(Nz,1);
C_lqr = zeros(Nz,1);
rhs = zeros(Nz,1);
for k=1:Nz
    C(k) = norm(inv(sqrtm(P))*[eye(nx), K']*L(k,:)')^2;
    C_lqr(k) = norm(inv(sqrtm(P_lqr))*[eye(nx), -K_lqr']*L(k,:)')^2;
    rhs(k) = (l(k)-L(k,:)*r)^2;
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

R = 1e-3

if writeout
    writematrix(reshape(P.',1,[]), 'mpc_parameters//P.txt')
    writematrix(reshape(Q.',1,[]), 'mpc_parameters//Q.txt')
    writematrix(reshape(K.',1,[]), 'mpc_parameters//K.txt')
    writematrix(reshape(R.',1,[]), 'mpc_parameters//R.txt')
    writematrix(alpha,             'mpc_parameters//alpha.txt')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Incrementalstability %%
%%%%%%%%%%%%%%%%%%%%%%%%%%

ue      = 0.7853
xe1     = 0.2632
xe2     = 0.6519

u_min = [0-ue];
u_max = [2-ue];

x_min = [-0.2,-0.2]
x_max = [0.2,0.2]
x_step = (x_max-x_min)/30
u_step = (u_max-u_min)/30

[X1, X2] = ndgrid(x_min:x_step:x_max);
[U] = ndgrid(u_min:u_step:u_max);
N_xgrid = numel(X1);
N_ugrid = numel(U);

disp("Grid size x")
disp(N_xgrid)
disp("Grid size u")
disp(N_ugrid)

%% system dynamics in casadi
x = SX.sym('x',nx);
u = SX.sym('u',nu);
f = Function('f', {x,u}, {stirtank_dynamics(x(1), x(2), u) } );
Afunc = Function('A', {x,u}, {jacobian(f(x,u), x)});
Bfunc = Function('B', {x,u}, {jacobian(f(x,u), u)});

%% setup variables for lmi optimization
con=[];
c_u=1;
rho_c=0.05;
wbar=1;
Y = sdpvar(nu,nx);
X = sdpvar(nx);
e = 0.0;
% mpc stage cost
Q = eye(nx);
R = 1e-1;

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

if writeout
    writematrix(wbarmin, 'mpc_parameters/wbar.txt')
    writematrix(rho_c, 'mpc_parameters/rho_c.txt')
    writematrix(reshape(Pdelta.',1,[]), 'mpc_parameters/Pdelta.txt')
    writematrix(reshape(Kdelta.',1,[]), 'mpc_parameters/Kdelta.txt')
end

cj = [];
for i = 1:size(Lx,1)
    cj = [cj; norm(Pdelta^-(1/2)*(Lx(i,:)'+Kdelta'*Lu(i,:)'),2)]
end

c_max = max(cj)

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