%4th order Runge-Kutta integration routine
clear,clc
%Parameters that was given in papers

% Don't change
rhoNa = 60;
rhoK = 18;
tsyn = 0.003;

% Can be changed
n_neurons = 5;
gNa = 120;
gK = 36;
gL = 0.3;
Cm = 1;
vNa = 115;
vK = -12;
vL = 10.6;
Iapp = 70;
Mem_sp = 0.1;  % in the paper 10^5 but we will not see the noize
global ge;
ge = 0.005;
Erev = 70;



% Routine starts here
Am = @(v) 0.1*(25-v)./(exp((25-v)./10)-1);
Bm = @(v) 4*exp(-v./18);

Ah = @(v) 0.07*exp(-v./20);
Bh = @(v) 1./(exp((30-v)./10)+1);

An = @(v) 0.01*(10-v)./(exp((10-v)./10)-1);
Bn = @(v) 0.125*exp(-v./80);


% Hope that I understand noize in a correct way;
dz_m = @(v) sqrt(2*Am(v).*Bm(v)./(rhoNa*Mem_sp*(Am(v)+Bm(v)))) .* randn(n_neurons,1);
dz_h = @(v) sqrt(2*Ah(v).*Bh(v)./(rhoNa*Mem_sp*(Ah(v)+Bh(v)))) .* randn(n_neurons,1);
dz_n = @(v) sqrt(2*An(v).*Bn(v)./(rhoK*Mem_sp*(An(v)+Bn(v)))) .* randn(n_neurons,1);

m_dot = @(v,m) Am(v).*(1-m) - Bm(v).*m;
h_dot = @(v,h) Ah(v).*(1-h) - Bh(v).*h;
n_dot = @(v,n) An(v).*(1-n) - Bn(v).*n;


v_dot = @(v,m,h,n,t,Isyn) (-gK.*n.^4.*(v - vK) -gNa.*m.^3.*h.*(v - vNa) ...
    -gL.*(v-vL) + Iapp + Isyn)./Cm;


step = 0.01;
t = 0:step:100;
v = zeros(n_neurons,length(t));
m = zeros(n_neurons,length(t));
h = zeros(n_neurons,length(t));
n = zeros(n_neurons,length(t));

% initial values
% v(1) = 5;
% m(1) = 0.08;
% h(1) = 0.6;
% n(1) = 0.33;

k1_v = zeros(1,n_neurons);
k1_m = zeros(1,n_neurons);
k1_h = zeros(1,n_neurons);
k1_n = zeros(1,n_neurons);

k2_v = zeros(1,n_neurons);
k2_m = zeros(1,n_neurons);
k2_h = zeros(1,n_neurons);
k2_n = zeros(1,n_neurons);

k3_v = zeros(1,n_neurons);
k3_m = zeros(1,n_neurons);
k3_h = zeros(1,n_neurons);
k3_n = zeros(1,n_neurons);


k4_v = zeros(1,n_neurons);
k4_m = zeros(1,n_neurons);
k4_h = zeros(1,n_neurons);
k4_n = zeros(1,n_neurons);

nz_m = dz_m(v(:,1));
nz_h = dz_h(v(:,1));
nz_n = dz_n(v(:,1));


for j=1:(length(t)-1)
    
    for i = 1:n_neurons
        Isyn = Isynapse(v(:,j), i);
        k1_v(i) = v_dot(v(i,j),m(i,j),h(i,j),n(i,j),t(j), Isyn);
    
        k1_m(i) = m_dot(v(i,j),m(i,j)) + nz_m(i);
        k1_h(i) = h_dot(v(i,j),h(i,j)) + nz_h(i);
        k1_n(i) = n_dot(v(i,j),n(i,j)) + nz_n(i);
    end
    
    nz_m = dz_m(v(:,j)+0.5*step.*transpose(k1_v));
    nz_h = dz_h(v(:,j)+0.5*step.*transpose(k1_v));
    nz_n = dz_n(v(:,j)+0.5*step.*transpose(k1_v));
    
    for i = 1:n_neurons
        Isyn = Isynapse(v(:,j)+0.5*step*k1_v, i);
        
        k2_v(i) = v_dot(v(i,j)+0.5*step*k1_v(i), m(i,j)+0.5*step*k1_m(i),...
            h(i,j)+0.5*step*k1_h(i), n(i,j)+0.5*step*k1_n(i),...
            t(j)+0.5*step, Isyn);

        k2_m(i) = m_dot(v(i,j)+0.5*step*k1_v(i),m(i,j)+0.5*step*k1_m(i)) + nz_m(i);
        k2_h(i) = h_dot(v(i,j)+0.5*step*k1_v(i),h(i,j)+0.5*step*k1_h(i)) + nz_h(i);
        k2_n(i) = n_dot(v(i,j)+0.5*step*k1_v(i),n(i,j)+0.5*step*k1_n(i)) + nz_n(i);
    end
    
    
    for i = 1:n_neurons
        Isyn = Isynapse(v(:,j)+0.5*step*k2_v, i);
        
        k3_v(i) = v_dot(v(i,j)+0.5*step*k2_v(i), m(i,j)+0.5*step*k2_m(i),...
            h(i,j)+0.5*step*k2_h(i), n(i,j)+0.5*step*k2_n(i),...
            t(j)+0.5*step, Isyn);

        k3_m(i) = m_dot(v(i,j)+0.5*step*k2_v(i), m(i,j)+0.5*step*k2_m(i)) + nz_m(i); 
        k3_h(i) = h_dot(v(i,j)+0.5*step*k2_v(i), h(i,j)+0.5*step*k2_h(i)) + nz_h(i);
        k3_n(i) = n_dot(v(i,j)+0.5*step*k2_v(i), n(i,j)+0.5*step*k2_n(i)) + nz_n(i);
    end
    
    nz_m = dz_m(v(:,j)+step.*transpose(k3_v));
    nz_h = dz_h(v(:,j)+step.*transpose(k3_v));
    nz_n = dz_n(v(:,j)+step.*transpose(k3_v));
    
    for i = 1:n_neurons
        Isyn = Isynapse(v(:,j)+step*k3_v, i);
        
        k4_v(i) = v_dot(v(i,j)+step*k3_v(i), m(i,j)+step*k3_m(i),...
            h(i,j)+step*k3_h(i), n(i,j)+step*k3_n(i),...
            t(j)+step, Isyn);

        k4_m(i) = m_dot(v(i,j)+step*k3_v(i), m(i,j)+step*k3_m(i))+nz_m(i);
        k4_h(i) = h_dot(v(i,j)+step*k3_v(i), h(i,j)+step*k3_h(i))+nz_h(i);
        k4_n(i) = n_dot(v(i,j)+step*k3_v(i), n(i,j)+step*k3_n(i))+nz_n(i);
    end
    
    
    for i = 1:n_neurons
        v(i,j+1) = v(i,j) + (1/6)*(k1_v(i) + 2*k2_v(i) +...
            2*k3_v(i) + k4_v(i))*step;
        
        m(i,j+1) = m(i,j) + (1/6)*(k1_m(i) + 2*k2_m(i) +...
            2*k3_m(i) + k4_m(i))*step;
        
        h(i,j+1) = h(i,j) + (1/6)*(k1_h(i) + 2*k2_h(i) +...
            2*k3_h(i) + k4_h(i))*step;
        
        n(i,j+1) = n(i,j) + (1/6)*(k1_n(i) + 2*k2_n(i) +...
            2*k3_n(i) + k4_n(i))*step;
    end
end


v_aver = mean(v);


figure(1)
plot(t,v,'-','LineWidth', 0.5, 'DisplayName','v')
hold on;
plot(t,v_aver,'-.','LineWidth', 1,'DisplayName','v_{aver}')
xlabel('t')
ylabel('v')
legend
hold off;
figure(2)
plot(t,m, t,h, t,n)
xlabel('t')
ylabel('cc')
legend('m','h','n')


C_up = @(tau, v) ((v(:, 1: end - tau) - v_aver(1, 1:end - tau))*...
    transpose(v(:,1+tau: end) - v_aver(1, 1+tau:end))).*...
    eye(n_neurons)*ones(n_neurons,1);

C_down = @(tau, v)  ((v(:,1: end) - v_aver(1,1: end))*...
    transpose(v(:,1: end) - v_aver(1,1: end))).*eye(n_neurons)*...
    ones(n_neurons,1);

C = @(tau, v) mean(C_up(tau, v))/mean(C_down(tau,v));



t_char = 0;
for tau=1:(length(t))
    t_char = t_char + C(tau,v).^2 * step;
end
t_char


