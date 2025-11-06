clear all;
clc;
addpath(genpath('./'));

dsPath='datasets/';
ds = {
'MSRC',...
'BBCSport',...
'Wiki_fea',...
'CCV',...
'BBC',...
'BDGP_fea',...
'caltech101_8677',...
'AWAfea',...%8
'MNIST_fea',...
'Caltech101-20',...%10
'Notting-Hill',...
'Hdigit1',...
'ALOI',...%13
'Animal',...
'YoutubeFace_sel',...
'synthetic3d'};


alpha =0.1; % log-sum para, fix to 0.1
max_iter = 50; 
tol = 0.01;%Convergence threshold

%data preprocessing methods, default "unit"
para_perp={'MM1','no','unit','BN','no','BN','unit','unit','no','BN','unit','unit','unit','unit','BN','no'};

for di=6:6
    dataName = ds{di}; 
    disp(dataName);
    load(strcat(dsPath,dataName));
    true_label = Y'; % true label
    clear Y;
    V = length(X); % number of view
    for v = 1:V
        X{v} = X{v}';
    end
    c = length(unique(true_label)); % number of class
    n = size(X{1}, 2); %number of samples
    
    perp_f=para_perp{di};
    [X] = data_prep(X,perp_f);%'BN' 'LN' 'MM1' 'MM2' 'unit' 'no'
    
    
    para_range=[0.001,0.01,0.1,1,10];
    
    for i_m=1:5
        for i_i=1:5
            for i_j=1:5      
             p=c*i_m;
             m=c*i_m;
             lambda = para_range(i_i); % 
             gamma = para_range(i_j); % 
           
             tic;
             [A,Z,F_n,obj] = algo(c, n, V, X, p, m, lambda, gamma, alpha , max_iter, tol);  
             obj_all=obj.obj_values;
             obj_Z0=obj.obj_mu2;
                
                res = myNMIACCwithmean(F_n,true_label,c);
                time(i_m,i_i,i_j)=toc;
                acc(i_m,i_i,i_j)=res(1);
                nmi(i_m,i_i,i_j)=res(2);
                purity(i_m,i_i,i_j)=res(3);
                fscore(i_m,i_i,i_j)=res(4);
    
                fprintf('acc:%.4f,nmi:%.4f,p:%d,m:%d,i:%d,j:%d\n', ...
                    acc(i_m,i_i,i_j),nmi(i_m,i_i,i_j),i_m,i_i,i_j);
                
                dataset_name=['para/',dataName,'-0.1',perp_f,'.mat'];
             save(dataset_name,"acc","nmi","purity","fscore",'time');
            end
        end
    end
   
    %find max_acc para
    [maxVal, linearIdx] = max(acc, [], 'all'); 
    [best_m,best_i, best_j] = ind2sub(size(acc), linearIdx); 
    fprintf('max_acc:%.4f,best_m:%d,best_lambda:%.4f,best_gamma:%.4f\n' ...
        ,maxVal,best_m*c,para_range(best_i),para_range(best_j));
    
     clear acc nmi purity fscore time;
end