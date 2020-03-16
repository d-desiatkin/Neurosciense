% Here lies the results of Runge_kutta_network experiment

% Dependency of t_char from Mem_sp (ge = 0.005, Iapp = 20)
Mem_sp = [10^5, 10^4, 10^3, 10^2, 10, 1, 0.1];
t_char = [1.3029, 1.3377, 1.4023, 1.7066, 2.5424, 2.6149, 1.8759];

figure(1)
semilogx(Mem_sp, t_char)

% Dependency of t_char from Mem_sp (ge = 0.05, Iapp = 20)
Mem_sp = [10^5, 10^4, 10^3, 10^2, 10, 1, 0.1, 0.01];
t_char = [0.8787, 0.5171, 0.6057, 0.9537, 0.6293, 0.9174, 1.1225, 1.0246];

figure(2)
semilogx(Mem_sp, t_char)


% Dependency of t_char from Mem_sp (ge = 0.005, Iapp = 70)
Mem_sp = [10^5, 10^4, 10^3, 10^2, 10, 1, 0.1];
t_char = [3.1272, 3.2313, 3.1807,  2.8503, 4.1514, 2.8389, 1.5758];

figure(3)
semilogx(Mem_sp, t_char)


% Dependency of t_char from Mem_sp (ge = 0.05, Iapp = 70)
Mem_sp = [];
t_char = [];


% figure(4)
% semilogx(Mem_sp, t_char)
