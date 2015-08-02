%% sample the data to mat_final cell array
% mat_final = 140*1 cell array. Every cell is a 1146*1 cell, corresponding
% to a certain trail, every trail contains a 1146 sample result,
% corresponding to 9 seconds.
sample_rev50 = 0;
mat_all = cell(140,1);
% index_rev = 0;

    index_rev = index_rev + 1;
    rev_value = 0;
    for i = 1 : 140
        disp(i/140);
        %first sample C3
        mat1 = cell(1146,1);
        [result_temp, sample_error_temp,sample_rev_temp]=aar(x_train(:,1,i),[1,2],[6,0],0.035,[0,0,0,0,0,0],0.1*eye(6));
%         rev_value = rev_value + sample_rev_temp;
        
        for j = 1:1146
            mat1{j}=result_temp(j+6,:);
        end
        
        %sample C4
        mat2 = cell(1146,1);
        [result_temp, sample_error_temp,sample_rev_temp]=aar(x_train(:,3,i),[1,2],[6,0],0.035,[0,0,0,0,0,0],0.1*eye(6));
%         rev_value = rev_value + sample_rev_temp;
        
        for j = 1:1146
            mat2{j}=result_temp(j+6,:);
        end
        
        %concatenate the sample matrix
        mat_con = cell(1146,1);
        for j = 1:1146
            mat_con{j} = [mat1{j} mat2{j}];
        end
        mat_all{i} = mat_con;
        
    end
    
    %calibrate the matrix to the certain order
    %first 70 elements are classified to 1, the following 70 are classified to 2
    mat_train = cell(140,1);
    index_1=find(y_train==1);
    index_2=find(y_train==2);
    index = 1;
    for j = index_1'
        mat_train{index} = mat_all{j};
        index = index+1;
    end
    for j = index_2'
        mat_train{index} = mat_all{j};
        index = index+1;
    end
    
    %% construct the target array
    y_target=[ones(70,1) zeros(70,1)
        zeros(70,1) ones(70,1)];
    
    %% sample the test data
    mat_all = cell(140,1);
    for i = 1:140
        disp(i/140);
        %first sample C3
        mat1 = cell(1146,1);
        [result_temp, sample_error_temp, sample_rev_temp]=aar(x_test(:,1,i),[1,2],[6,0],0.035,[0,0,0,0,0,0],0.1*eye(6));
%         rev_value = rev_value + sample_rev_temp;
        for j = 1:1146
            mat1{j}=result_temp(j+6,:);
        end
        
        %sample C4
        mat2 = cell(1146,1);
        [result_temp, sample_error_temp, sample_rev_temp]=aar(x_test(:,3,i),[1,2],[6,0],0.035,[0,0,0,0,0,0],0.1*eye(6));
%         rev_value = rev_value + sample_rev_temp;
        for j = 1:1146
            mat2{j}=result_temp(j+6,:);
        end
        
        %concatenate the sample matrix
        mat_con = cell(1146,1);
        for j = 1:1146
            mat_con{j} = [mat1{j} mat2{j}];
        end
        mat_all{i} = mat_con;
        
    end
%     sample_rev50 = rev_value;


mat_test = mat_all;

%% transform data ready for LDA and QDA

mat_training = cell(1146,1);
mat_testing = cell(1146,1);
for j = 1:1146
    mat_temp_1 = cell(140,1);
    mat_temp_2 = cell(140,1);
    for i = 1:140
        mat_temp_1{i} = mat_train{i}{j};
        mat_temp_2{i} = mat_test{i}{j};
    end
    mat_training{j} = cell2mat(mat_temp_1);
    mat_testing{j} = cell2mat(mat_temp_2);
end

%% LDA&QDA
[accuracy_LDA] = da(mat_training, mat_testing, [ones(70,1);2*ones(70,1)]', y_test',1);
[accuracy_QDA] = da(mat_training, mat_testing, [ones(70,1);2*ones(70,1)]', y_test',2);

%% train the network by the array
result_test = cell(1146,1);
error = cell(1146,1);
for i = 1:1146
    mat_train_single = cell(140,1);
    mat_test_single = cell(140,1);
    for j = 1:140
        mat_train_single{j} = mat_train{j}{i};
        mat_test_single{j} = mat_test{j}{i};
    end
    [output, error_temp] = NN_test1(reshape(cat(3, cell2mat(mat_train_single)),[140 12]), y_target, reshape(cat(3, cell2mat(mat_test_single)),[140 12]));
    result_test{i} = output;
    error{i} = error_temp;
end

%% compare the result of NN to test target
accuracy_nn = cell(1146,1);
for i = 1:1146
    accuracy_nn{i} = 1 - sum(y_test ~= result_test{i}')/140;
end


