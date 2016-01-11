function predict_label = test_svm(featLength)

addpath(genpath('~/Downloads/liblinear-2.1'))
%addpath(genpath('~/Research/liblinear-1.96'));

%feature getpid
% input 
filename = 'train.mat';
load(filename);
filename = 'dev.mat';
load(filename);

%X_train = train_data(:,1:featLength);
%Y_train = train_data(:,featLength+1:featLength+2);
X_test = dev(:,1:featLength);
%Y_test = dev(:,101:102);

for i = 1:1
%train_label = Y_train(:,i);
% test_label = Y_test(:,i);

%sc = 1e0
%model = train(train_label, sparse(X_train), '-s 0 -c 1e0 -B 1 -q');
load model.mat
[predict_label, accuracy, dec_values] = predict(zeros(size(X_test,1),1), sparse(X_test), model);

% build confusion matrix.
% label = unique(test_label);
% num_class = length(label);
% num_sample = size(test_label,1);
% confusion = zeros(num_class);
% 
% for j = 1:num_sample
%     confusion(test_label(j)+1,predict_label(j)+1) = confusion(test_label(j)+1,predict_label(j)+1) + 1;
% end
% 
% acc(i).precision = precision(confusion);
% acc(i).recall = recall(confusion);

% acc(i) = sum(predict_label==test_label)/length(test_label)

end
% save('acc.mat','acc');

end
