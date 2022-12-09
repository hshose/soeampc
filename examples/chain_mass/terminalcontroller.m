%continous-time formulation
clear all;
clc;

addpath('/home/hose/software/casadi')
import casadi.*
addpath('./dynamics/');

rng('default');
rng(42);

for n_mass = 3:6
    % n_mass = 6;          % number of masses
    M = n_mass - 2;      % number of intermediate masses
    nx = (2*M + 1)*3;    % differential states
    nxpos = (M+1)*3;
    nxvel = M*3;
    nu = 3;              % control inputs


    xend = [1;0;0; zeros(nxvel,1)];
    xinit = zeros(3*M, 1);
    for i=1:M
        x1init = linspace(0,xend(1),n_mass);
        x2init = linspace(0,xend(2),n_mass);
        x3init = linspace(0,xend(3),n_mass);
        xinit((i-1)*3+1,1) = x1init(i+1);
        xinit((i-1)*3+2,1) = x2init(i+1);
        xinit((i-1)*3+3,1) = x3init(i+1);
    end
    % disp(faut(xinit))
    disp("xinit")
    disp(xinit)
    xrefpos = fsolve(@(x) faut(x, xend), xinit);
    disp("xrefpos")
    disp(xrefpos)
    xref = [xrefpos;xend];
    disp("xref")
    disp(xref)

    x = SX.sym('x',nx);  % states
    u = SX.sym('u',nu);  % inputs

    fexpl = Function('fexpl',{x,u},{f(x+xref,u)});                 % explicit ct dynamics
    Afunc = Function('A', {x,u}, {jacobian(fexpl(x,u), x)});    % ct linearization
    Bfunc = Function('B', {x,u}, {jacobian(fexpl(x,u), u)});    % ct linearization
    % usage:
    %   A = full(Afunc(xval, uval))
    %   B = full(Bfunc(xval, uval))

    %% setup variables for lmi optimization
    con=[];
    % rho_c=0.192;
    % rho_c=0.0802;
    rho_c=0.192;
    Y = sdpvar(nu,nx);
    X = sdpvar(nx);
    % mpc stage cost
    % q = ones(nx,1);
    % q(3*M+1) = M+1;
    % q(3*M+2) = M+1;
    % q(3*M+3) = M+1;

    q = [ones(3*M,1); 25*ones(3,1); ones(nxvel,1)];
    Q = diag(q);
    R = 1e-2*eye(nu);
    e=0;

    % random samples
    Nsamples = 1e3;
    xmin = -ones(nx,1);
    xmax = ones(nx,1);
    nbx = M+1;
    xmin(1:nxpos) = -0.1;
    xmax(1:nxpos) =  0.1;
    umin = -ones(nu,1);
    umax = ones(nu,1);

    %% iterate over grid and add lmi constraints
    for i = 1:Nsamples
        px = xmin + rand(nx,1).*(xmax-xmin);
        pu = umin + rand(nu,1).*(umax-umin);
        A = full(Afunc(px, pu));
        B = full(Bfunc(px, pu));
        % disp(A)
        % disp(B)
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

    A = full(Afunc(zeros(nx,1), zeros(nu,1)));
    B = full(Bfunc(zeros(nx,1), zeros(nu,1)));
    [K_lqr,P_lqr] = lqr(A,B,Q+e*eye(nx),R);

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

    nconstr = (M+1) + 2*nu

    Lx = zeros(nconstr,nx);
    Lu = zeros(nconstr,nu);
    l = ones(nconstr,1);

    umax = 1;
    ywall = -0.1;
    for i=1:M+1
        Lx(i,(i-1)*3+2) = 1/ywall;
    end
    for i=1:nu
        Lu(M+1+i,i)=1/umax;
        Lu(M+1+nu+i,i)=1/-umax;
    end
    disp('Lx')
    disp(Lx)
    disp('Lu')
    disp(Lu)


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


    writematrix(reshape(round(P,6),1,[]),   'mpc_parameters/P_'+string(n_mass)+'.txt');
    writematrix(reshape(round(Q,6),1,[]),   'mpc_parameters/Q_'+string(n_mass)+'.txt');
    writematrix(reshape(round(K,6),1,[]),   'mpc_parameters/K_'+string(n_mass)+'.txt');
    writematrix(reshape(round(R,6),1,[]),   'mpc_parameters/R_'+string(n_mass)+'.txt');
    writematrix(alpha,                      'mpc_parameters/alpha_'+string(n_mass)+'.txt');
    writematrix(reshape(round(xref,6),1,[]), 'mpc_parameters/xref_'+string(n_mass)+'.txt');
end