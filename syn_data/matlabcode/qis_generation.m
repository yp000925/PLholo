function out = qis_generation(params, IM)

K = params.K;               % Spatial  Overasampling Factor
T = params.T;                % Temporal Overasampling Factor
Qmax = parmas.Qmax;              % Maximum Threshold
g  =  ones(K)/K^2;      % Box-car Interpolation kernel

[rows, cols] = size(IM);
alpha = K^2*(qmax-1);
y =  generatePhotonCounts(IM,K,T,alpha);
Qmap =  ones(size(IM))*Qmax; % [qmax]
oracleQmap_HR=kron(Qmap,ones(K));
b =  1*(y>=repmat(oracleQmap_HR,1,1,T));
end
