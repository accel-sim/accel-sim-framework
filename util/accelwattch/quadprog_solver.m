%Copyright (c) 2018-2021, Vijay Kandiah, Junrui Pan, Mahmoud Khairy, Scott Peverelle, Timothy Rogers, Tor M. Aamodt, Nikos Hardavellas
%Northwestern University, Purdue University, The University of British Columbia
%All rights reserved.
%
%Redistribution and use in source and binary forms, with or without
%modification, are permitted provided that the following conditions are met:
%
%1. Redistributions of source code must retain the above copyright notice, this
%   list of conditions and the following disclaimer;
%2. Redistributions in binary form must reproduce the above copyright notice,
%   this list of conditions and the following disclaimer in the documentation
%   and/or other materials provided with the distribution;
%3. Neither the names of Northwestern University, Purdue University,
%   The University of British Columbia nor the names of their contributors
%  may be used to endorse or promote products derived from this software
%   without specific prior written permission.
%
%THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
%AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
%IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
%ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
%LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
%CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
%SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
%INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
%CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
%ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
%POSSIBILITY OF SUCH DAMAGE.

input = csvread('accelwattch_volta_sass_sim.csv');
A = input(:,1:31); % change 30 to number of power counters if different
b = input(:,32);
l = 0.1*ones(1,31); % lower bounds
u = 1000*ones(1,31); % upper bounds

M= zeros(1,31);
N= [0];



C = zeros(16,31);
D = zeros(16,1);
%These factors are calculated using McPAT per instruction energies * current AccelWattch scaling factors. They need to be changed after each iteration of the solver + simulator runs.
C(1,8)=1;
C(1,9)=-1.843582172; %INT <= FPU

C(2,9)=1;
C(2,10)=-0.999991619; %FPU <= DPU

C(3,8)=1;
C(3,11)=-1.107050869; %INT <= INT_MUL24

C(4,8)=1;
C(4,12)=-1.000003101; %INT <= INT_MUL32

C(5,8)=1;
C(5,13)=-1.00000271; %INT <= INT_MUL

C(8,15)=1;
C(8,16)=-14.17176203; %FP_MUL <= FP_DIV

C(9,15)=1;
C(9,21)=-1.063781571; %FP_MUL <= DP_MUL

C(10,15)=1;
C(10,17)=-5.587170605; %FP_MUL <= FP_SQRT

C(11,15)=1;
C(11,18)=-2.082920111; %FP_MUL <= FP_LG

C(12,15)=1;
C(12,19)=-1.767684722; %FP_MUL <= FP_SIN

C(13,15)=1;
C(13,20)=-1.438999757; %FP_MUL <= FP_EXP

C(15,15)=1;
C(15,23)=-75.07256801; %FP_MUL <= TENSOR

C(16,15)=1;
C(16,24)=-0.999997535; %FP_MUL <= TEXP

%The Idle_Core_power, static power, constant power components are already
%modeled prior to dynamic power quadprog optimization
l(29)=1;
u(29)=1;
l(30)=1;
u(30)=1;
l(31)=1;
u(31)=1;

result = quadprog(2*A'*A, -2*A'*b, C, D, [], [], l, u);
csvwrite('scaled_coefficients.csv', result);
