function f = quadrotor_dynamics(x, u)
    d0 = 10;
    d1=8;
    n0=10;
    kT=0.91;
    g=9.8;
    f = [x(4), x(5), x(6), ...
        g*tan(x(7)), g*tan(x(9)), -g + kT*u(3),...
        -d1*x(7)+x(8), -d0*x(7)+n0*u(1),...
        -d2*x(9)+x(10), -d0*x(9)+n0*u(2)
        ];
end