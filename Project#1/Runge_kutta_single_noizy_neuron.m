%4th order Runge-Kutta integration routine
clear,clc
%Parameters that was given in papers

% Don't change
rhoNa = 60;
rhoK = 18;
tsyn = 0.003;

% Can be changed
gNa = 120;
gK = 36;
gL = 0.3;
Cm = 1;
vNa = 115;
vK = -12;
vL = 10.6;
Iapp = 20;
Mem_sp = 0.1;  % in the paper 10^5 but we will not see the noize
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
dz_m = @(v) sqrt(2*Am(v).*Bm(v)./(rhoNa*Mem_sp*(Am(v)+Bm(v)))) * randn(1);
dz_h = @(v) sqrt(2*Ah(v).*Bh(v)./(rhoNa*Mem_sp*(Ah(v)+Bh(v)))) * randn(1);
dz_n = @(v) sqrt(2*An(v).*Bn(v)./(rhoK*Mem_sp*(An(v)+Bn(v)))) * randn(1);

m_dot = @(v,m) Am(v).*(1-m) - Bm(v).*m;
h_dot = @(v,h) Ah(v).*(1-h) - Bh(v).*h;
n_dot = @(v,n) An(v).*(1-n) - Bn(v).*n;


v_dot = @(v,m,h,n,t,vn) (-gK.*n.^4.*(v - vK) -gNa.*m.^3.*h.*(v - vNa) ...
    -gL.*(v-vL) + Iapp)./Cm;


step = 0.01;
t = 0:step:100;
v = zeros(1,length(t));
m = zeros(1,length(t));
h = zeros(1,length(t));
n = zeros(1,length(t));
v_j = zeros(1,length(t));
m_j = zeros(1,length(t));
h_j = zeros(1,length(t));
n_j = zeros(1,length(t));
% initial values
v(1) = 5;
m(1) = 0.08;
h(1) = 0.6;
n(1) = 0.33;

nz_m = dz_m(v(1));
nz_h = dz_h(v(1));
nz_n = dz_n(v(1));


for i=1:(length(t)-1)
    
    k1_v = v_dot(v(i),m(i),h(i),n(i),t(i));
    
    k1_m = m_dot(v(i),m(i)) + nz_m;
    k1_h = h_dot(v(i),h(i)) + nz_h;
    k1_n = n_dot(v(i),n(i)) + nz_n;
    
    nz_m = dz_m(v(i)+0.5*step*k1_v);
    nz_h = dz_h(v(i)+0.5*step*k1_v);
    nz_n = dz_n(v(i)+0.5*step*k1_v);
    
    k2_v = v_dot(v(i)+0.5*step*k1_v, m(i)+0.5*step*k1_m,...
        h(i)+0.5*step*k1_h, n(i)+0.5*step*k1_n, t(i)+0.5*step);
    
    k2_m = m_dot(v(i)+0.5*step*k1_v,m(i)+0.5*step*k1_m)+nz_m;
    k2_h = h_dot(v(i)+0.5*step*k1_v,h(i)+0.5*step*k1_h)+nz_h;
    k2_n = n_dot(v(i)+0.5*step*k1_v,n(i)+0.5*step*k1_n)+nz_n;
    
    
    k3_v = v_dot(v(i)+0.5*step*k2_v, m(i)+0.5*step*k2_m,...
        h(i)+0.5*step*k2_h, n(i)+0.5*step*k2_n, t(i)+0.5*step);
    
    k3_m = m_dot((v(i)+0.5*step*k2_v),(m(i)+0.5*step*k2_m))+nz_m; 
    k3_h = h_dot((v(i)+0.5*step*k2_v),(h(i)+0.5*step*k2_h))+nz_h;
    k3_n = n_dot((v(i)+0.5*step*k2_v),(n(i)+0.5*step*k2_n))+nz_n;
    
    nz_m = dz_m(v(i)+step*k3_v);
    nz_h = dz_h(v(i)+step*k3_v);
    nz_n = dz_n(v(i)+step*k3_v);

    k4_v = v_dot(v(i)+step*k3_v, m(i)+step*k3_m,...
        h(i)+step*k3_h, n(i)+step*k3_n, t(i)+step);
    
    k4_m = m_dot((v(i)+step*k3_v),(m(i)+step*k3_m)) + nz_m;
    k4_h = h_dot((v(i)+step*k3_v),(h(i)+step*k3_h)) + nz_h;
    k4_n = n_dot((v(i)+step*k3_v),(n(i)+step*k3_n)) + nz_n;
    
    
    v(i+1) = v(i) + (1/6)*(k1_v + 2*k2_v + 2*k3_v + k4_v)*step;
    m(i+1) = m(i) + (1/6)*(k1_m + 2*k2_m + 2*k3_m + k4_m)*step;
    h(i+1) = h(i) + (1/6)*(k1_h + 2*k2_h + 2*k3_h + k4_h)*step;
    n(i+1) = n(i) + (1/6)*(k1_n + 2*k2_n + 2*k3_n + k4_n)*step;
    
end



figure(1)
plot(t,v)
xlabel('t')
ylabel('v')
figure(2)
plot(t,m, t,h, t,n)
xlabel('t')
ylabel('cc')
legend('m','h','n')

dv = v_dot(v,m,h,n,t,v_j);
figure(3)
plot(dv,v)
xlabel('dv(t)')
ylabel('v(t)')

dm = m_dot(v,m);
figure(4)
plot(dm,m)
xlabel('dm(t)')
ylabel('m(t)')


