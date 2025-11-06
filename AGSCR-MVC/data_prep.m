function [X] = data_prep(X,method)
%
%Input:
%       X: cell类型,数据矩阵, size(X{i})=d*n;
%       method: 字符串类型;
%               'BN': Batch Normalization;
%               'LN': Layer Normalization;
%               'MM1': MinMax归一化至[0,1]
%               'MM2': MinMax归一化至[-1,1]
v = length(X);
% [d,n] = size(X{1});
switch method
    case 'BN'
        for i = 1:v
            X{i} = mapstd(X{i},0,1); % 数据预处理--Batch Norm
        end
    case 'LN'
        for i = 1:v
            X{i} = transpose(mapstd(X{i}',0,1)); % 数据预处理--Layer Norm
        end
    case 'MM1'
        for i = 1:v
            dist = max(max(X{i})) - min(min(X{i}));
            X{i} = (X{i} - min(min(X{i})))/dist;  % MinMax归一化至[0,1]
        end
    case 'MM2'
        for i = 1:v
            dist = max(max(X{i})) - min(min(X{i}));
            m01 = (X{i} - min(min(X{i})))/dist;
            X{i} = 2 * m01 - 1;  % MinMax归一化至[-1,1]
        end
    case 'unit'
        for i = 1:v
            X{i} = X{i} ./ (repmat(sqrt(sum(X{i}.^2,1)),size(X{i},1),1)+eps);  % 单位向量归一化
        end
    case 'no'
        X = X;  % 不预处理
end


end

