function fexpl = f(x,u)
    
    M = ( length(x)/3 - 1 ) / 2;

    nxpos = (M+1)*3;
    nxvel = M*3;

    xpos = x(1:nxpos);
    xvel = x(nxpos+1:end);

    force = zeros(3*M,1, 'like', x);

    for i = 1:M
        force(3*i) = - 9.81;
    end

    % disp(force)

    % parameters hardcoded...
    L = 0.033;
    D = 1.0;
    m = 0.033;

    for i = 0:M
        if i == 0
            dist = xpos(i*3+1:(i+1)*3);
        else
            dist = xpos(i*3+1:(i+1)*3) - xpos((i-1)*3+1:i*3);
        end

        scale = D/m*(1-L/ norm(dist,2));
        F = scale*dist;

        % mass on the right
        if i < M
            force(i*3+1:(i+1)*3) = force(i*3+1:(i+1)*3) - F;
        end

        % mass on the left
        if i > 0
            force((i-1)*3+1:i*3) = force((i-1)*3+1:i*3) + F;

        end

    end

    fexpl = [xvel; u; force];

end