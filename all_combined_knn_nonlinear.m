%knn approach with active learning
% select two initial point
clear all
close all
clc

%% loading data

 
%    num =  xlsread('Australian');
%     num =  xlsread('Heart-Statlog');
%    num =  xlsread('Bupa-Liver');
%  x = num(:,2:end);
%  y = num(:,1);
%  y(find(y==-1),:)=2;

% load('sonar_data.mat')
% load('sonar_label.mat')
% x=data;
% y=y1;
% 
% load ionosphere_data.mat
% load ionosphere_label.mat
% x=data;
% y=y2;

% load wpbc_data.mat
% load wpbc_label.mat
% x=data;
% % 
load wdbc_data.mat
load label_wdbc.mat
y=y2;

% 
% load diabetes_data.mat
% load diabetes_label.mat
% x=data1;
% y=label;
% y(find(y==0),:)=2;

% 
% load echocardiogram_data.mat
% load echocardiogram_label.mat
% y(find(y==0),:)=2;

% % 
% load heart.dat
% x = heart(:,1:end-1);
% y = heart(:,end);

%%
x=datanorm_p(x);
m=5;
times=0;
C1=0.1;
kernel_para=.3;
rng(11)
K=10; % number of folds
%% Crossfold data
indices = crossvalind('Kfold',y,K);

for o=1:K
    disp('Fold no.');
    disp(o);
    test1=find(indices==o);
    train1=find(indices~=o);
    
    test_data1=x(test1,:);
    test_label1=y(test1,:);
    
    train_data1=x(train1,:);
    train_label1=y(train1,:);
    tic;
    [w1,b1,w2,b2,alpha,beta]=classifier1(train_data1,train_label1,C1, 2, kernel_para);%classifier acc to train data
    times=times+toc;
    test_data11=evalkernel(test_data1,train_data1,'rbf',kernel_para);
    
    hyper2(:,1)=abs(((test_data11*w1))+b1)/norm(w1);
    hyper2(:,2)=abs(((test_data11*w2))+b2)/norm(w2);
    
    for q=1:size(test_data11,1)
        [~,l1(q)]=min(hyper2(q,:));
    end
    
    count1=nnz(find(l1' == test_label1));
    n1=size(test_label1,1);
    accuracy1(o)= (count1*100)/n1;
    t(o)=times;
    
    clear  l1 hyper2 w1 b1 w2 b2 count1
    
end

fprintf('\n\nMean of benchmark Testing accuracy: %d',mean(accuracy1));
benchmark=mean(accuracy1);
fprintf('\n Standard Deviation of benchmark: %d\n',std(accuracy1));
disp(' Mean of CPU times: ');
disp(mean(t));
%% main algorithm starts here
trueacc=100;
no_query=0;

while (trueacc>3)
    no_query=no_query+1;
    for o=1:K
        disp(' Fold no.');
        disp(o);
        
        test=find(indices==o);
        train=find(indices~=o);
        
        test_data=x(test,:);
        test_label=y(test,:);
        
        train_data=x(train,:);
        train_label=y(train,:);
    
    
    knn2=[];
    knn21=[];
    
    %% finding the two classes

       class_1=find(train_label==1);
       class_2=find(train_label==2);  
    
    %% grouping the data according label class
    group1=train_data(class_1,:);
    group2=train_data(class_2,:);
    
    %% Randomly chosen 2 poits from whole data set
    ran1 = randi([1 size(group1,1)],1,1);
    ran2 =randi([1 size(group2,1)],1,1);
    
    chosen_point=[class_1(ran1); class_2(ran2)];
    %% two initial data randomnly chosen out of 690
     int_data1=train_data(chosen_point(1,:),:);
     int_data2=train_data(chosen_point(2,:),:);

    %% first 2 data point delete from original data number(690-2=688)
    final_indx=[1:length(train_data)]';
    unlabel_indx=setdiff(final_indx,chosen_point);
    total_data1=train_data(unlabel_indx,:);
    
    %% all distances of all data(98) from these two initial points
    
    dist1=pdist2(total_data1,int_data1);
    [dist12,ind1]=sort((dist1));
    
    dist2=pdist2(total_data1,int_data2);
    [dist22,ind2]=sort((dist2));
    
    %% taking five 1-knn
    k_point1=ind1(1:5,:);
    f_data1=train_data(k_point1,:); 
    
    k_point2=ind2(1:5,:);
    f_data2=train_data(k_point2,:);
   
    
    %% Finding the five-2nn points and its labels
    for i=1:size(f_data1,1)
        knn2(i,:)=pdist2(total_data1,f_data1(i,:));
        [distance1,ind11]=sort(knn2(i,:));
        two_nnpoint_index(i)=ind11(2);
        two_nnpoint(i,:)=total_data1(two_nnpoint_index(i),:);

    end
    
    %%  deleting 2nd 1nn data+five-2nn points from rest of data (688-5=683)
 
    final_indx=[1:size(total_data1,1)]';
    unlabel_indx2=setdiff(final_indx,two_nnpoint_index);
    unlabel_data1=train_data(unlabel_indx2,:);
    
    %%
    for i=1:size(f_data2,1)
        knn21(i,:)=pdist2(unlabel_data1,f_data2(i,:));
        [distance2,ind12]=sort(knn21(i,:));
        two_nnpoint_index2(i)=ind12(2);
        two_nnpoint2(i,:)=unlabel_data1(two_nnpoint_index2(i),:);
       
    end
    
    %%
    final_indx1=[1:size(total_data1,1)]';
    unlabel_indx2=setdiff(final_indx1,two_nnpoint_index2);
    unlabel_data2=train_data(unlabel_indx2,:);
    
    %%
    input_data_index=union(two_nnpoint_index,two_nnpoint_index2);
    unlabel_data_index=setdiff(unlabel_indx,input_data_index);
    
    input_data=train_data(input_data_index,:);
    data_label=train_label(input_data_index);
    
    one=find(data_label==1);
    two=find(data_label==2);
    one_size=size(one,1);
    two_size=size(two,1);
    if((one_size==0)||(two_size==0))
        [input_data,data_label]=random_generator(group1,group2,class_1,class_2,train_data,train_label);
    end
    
    unlabel_sample=train_data(unlabel_data_index,:);
    unlabel_group=train_label(unlabel_data_index);
    
    
    %% Function calling
    [W1,W2,B1,B2,query,times,chosendata]=max_query_all_combined_knn_nonlinear(no_query,input_data,data_label,unlabel_sample,unlabel_group,C1,kernel_para);
    
    g=test_acc_all_combined_knn_nonlinear(o,test_data,test_label,W1,W2,B1,B2,query,kernel_para);
    
    %%
    accuracy2(o)=g;
    tt(o)=times;
    cd(o)=chosendata;
    
    end
    mean_test=mean(accuracy2)
    fprintf('Mean Testing accuracy: %d',mean_test);
    fprintf('\n Standard Deviation : %d \n',std(accuracy2));
    disp(' Mean of CPU times: ');
    disp(mean(tt));
    disp('Mean of number of Actively Chosen data');
    disp(round(mean(cd)));
    disp(no_query);
    trueacc=benchmark-mean_test;
%         trueacc=-benchmark+mean_test;

end