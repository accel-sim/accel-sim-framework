input = csvread('volta_sass_sim.csv');
A = input(:,1:31); % change 30 to number of power counters if different
b = input(:,32);
l = 0.1*ones(1,31); % lower bounds
u = 1000*ones(1,31); % upper bounds

M= zeros(1,31);
N= [0];

C = zeros(16,31);
D = zeros(16,1);
% We don't model idle SM power, static power, and constant power here.
l(29)=1; 
u(29)=1;
l(30)=1;
u(30)=1;
l(31)=1;
u(31)=1;

result = quadprog(2*A'*A, -2*A'*b, C, D, [], [], l, u);
csvwrite('scaled_coefficients.csv', result);