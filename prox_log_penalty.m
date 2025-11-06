function Z0 = prox_log_penalty(Z, Y, alpha, mu)
% 逐元素计算log-sum惩罚邻近算子
%min alpha*Z0_log+mu/2(Z+Y/mu-Z0)

v = Z + Y./mu;
[m,n] = size(Z);
Z0 = zeros(m,n);

for i = 1:m
    for j = 1:n
        vij = v(i,j);
        disc = (1+vij)^2 - 4*alpha/mu;
        if disc >= 0
            zhat = (vij - 1 + sqrt(disc))/2;     
            % 
            zhat = max(zhat, 0);
            % 
            Z0(i,j) = zhat;
        else
            Z0(i,j) = 0;
        end
    end
end
end
