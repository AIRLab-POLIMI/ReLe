function dy = uwv_ode(t, y, u)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute ODE of Underwater Vehicle model
%----------------------------------------
v = y(1);
abs_v = abs(v);
c_v   = 1.2 + 0.2 * sin(abs_v);
m_v   = 3.0 + 1.5 * sin(abs_v);
k_v_u = -0.5 * tanh((abs(c_v * v * abs_v - u) - 30.0) * 0.1) + 0.5;
dy(1) = (u * k_v_u - c_v * v * abs_v) / m_v;
end
