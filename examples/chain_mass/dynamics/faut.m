function fexpl = faut(xMpos, xend)

    x = [xMpos; xend];
    u = zeros(3,1);
    fexpl = f(x,u);
end