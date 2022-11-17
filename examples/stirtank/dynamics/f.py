def f(x1, x2, u):
    theta   = 20
    k       = 300
    M       = 5
    xc      = 0.3816
    xf      = 0.3947
    alpha   = 0.117
    e       = 2.71828182845904523536028747135266249775724709369995
    xe1     = 0.2632
    xe2     = 0.6519
    ue      = 0.7853
    x1_dot = (1/theta)*(1-(x1+xe1))-k*(x1+xe1)*e**(-M/(x2+xe2))
    x2_dot = (1/theta)*(xf-(x2+xe2))+k*(x1+xe1)*e**(-M/(x2+xe2)) - alpha*(u+ue)*((x2+xe2)-xc)
    return x1_dot, x2_dot