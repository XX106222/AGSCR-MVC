function [A,Z,F_n,obj] = algo(c, n, V, X, p, m, lambda, gamma, alpha , max_iter, tol)


    %% Initialize
    W = cell(V, 1);
    for v = 1:V  
        di = size(X{v},1); 
        W{v} = zeros(di, p); 
    end

    A=eye(p,m);

    Z=zeros(m,n);
    Z(:,1:m) = eye(m); 
    %Z=ones(m,n)/m;
    F = eye(n+m, c) ;
   

    d_Z2 = sum(Z, 2);

    % ADMM para
    mu=1;
    rho =1.1;
    Z_0 = Z ;
    Y=zeros(m,n);

    ZZT = Z * Z';
    D_ZZT = sum(ZZT, 1) + 1e-6;
    D_ZZT_half = diag(D_ZZT.^(-0.5));
    L_A = eye(m) - D_ZZT_half * ZZT * D_ZZT_half;
    
    for iter = 1:max_iter
        %% Updata  W^(v)
        ZAT=Z' * A';
        parfor v = 1:V
            M_v = X{v} * ZAT;
            [U_W, ~, V_W] = svd(M_v, 'econ');
            W{v} = U_W * V_W';
        end
        
        %% Updata A

        R = zeros(p, m);
        for v = 1:V
            R = R + W{v}' * X{v} * Z';
        end
        M = V * ZZT + gamma  * L_A;
        A = R  / M;
        
 

        
        %% Updata Z
        
        ATA = A'*A;
        P1 = V* ATA;
        P2 = gamma*D_ZZT_half*ATA*D_ZZT_half;
      
        Q1 = zeros(m, n);
        for v = 1:V
            Q1 = Q1 + A' * W{v}' * X{v};
        end
        F_n = F(1:n, :); 
        F_m = F(n+1:end, :);
        
        D_Z2_half = diag(d_Z2.^(-0.5));
        
        Q2 = lambda  * F_n * F_m' * D_Z2_half;
        Q = Q1 + Q2'+ (mu/2) * Z_0-Y/2 ;
        
        Z = pinv(P1-P2+ (mu/2) * eye(m)) * Q; %  Z = P^{-1} Q
       
        parfor j = 1:n
            Z(:, j) = EProjSimplex_new(Z(:, j), 1);
        end
        
        Z=Z+eye(m,n)*1e-8;



      %% Updata F
        d_Z2 = diag(sum(Z, 2)); 
        D_Z2_half = diag(1./sqrt(diag(d_Z2)+1e-8));
        X_temp = Z' * D_Z2_half;
        [U_F, ~, V_F] = svd(X_temp,"econ"); 
        F_n = sqrt(2)/2 * U_F(:,1:c);
        F_m = sqrt(2)/2 * V_F(:,1:c);
        F = [F_n; F_m];

      


        %% Updata Z_0
        Z_0 =  prox_log_penalty(Z, Y, alpha, mu);
        
        %% Updata Y
        Y = Y + mu * (Z - Z_0);
        mu = min(rho * mu,1e4);
        

        %% obj_value

       %L_A
        ZZT = Z * Z'; % m x m
        D_ZZT=sum(ZZT,1)+1e-6;
        D_ZZT_half=diag(D_ZZT.^(-0.5));
        L_A=eye(m)-D_ZZT_half*ZZT*D_ZZT_half;

        obj.obj_data(iter) = 0;
        for v = 1:V
            obj.obj_data(iter) = obj.obj_data(iter) + norm(X{v} - W{v} * A * Z, 'fro')^2;
        end
        obj.obj_reg1(iter) = -2*lambda * trace(F_n'*X_temp*F_m);
        obj.obj_reg2(iter) = gamma * trace(A * L_A * A'); 
        obj.obj_Zlog(iter) = alpha*sum(sum(log1p(Z)));
  
        obj.obj_values(iter) = obj.obj_data(iter) + obj.obj_reg1(iter) + obj.obj_reg2(iter)+obj.obj_Zlog(iter);    
        %+0*obj_res_p(iter)
        
        obj.obj_mu2(iter)=mu*norm(Z - Z_0, 'fro')^2;

        fprintf('iter:%d,  all_obj:%f\n', ...
                iter,obj.obj_values(iter));
        
        if iter > 1 && abs(obj.obj_values(iter) - obj.obj_values(iter-1)) < tol
            break;
        end
    end
    F_n = F(1:n, :);
end