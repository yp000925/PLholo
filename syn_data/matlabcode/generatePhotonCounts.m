function y = generatePhotonCounts(c,K,T,alpha)
% Box-car kernel
theta   =  alpha*kron(c,ones(K))/K^2;  % Light Exposure % spatial oversample-> 每个jot 的range 为[0,qmax-1]
y       =  poissrnd(repmat(theta,1,1,T));  % Photon Count
 