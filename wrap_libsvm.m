function d = wrap_libsvm(C,F)
% wrapper to prep data & run libsvm 
%
% - Inputs are matrices for control [C] & patient [F] group.
% - C(ns x nv) where ns = subjects and nv is the variables to use
%
% * ensure you have DoPyTrain.py in libsvm-3.21/python directory
% * ensure convert.c is in the pwd
% * ensure libsvm-3.21/ directory is in pwd 
%
% -uses 'convert.c' to convert matrices to libsvm format data
% and compiles it for you if .c present.
% -writes a textfile in libsvm formatting called 'formd'
% -runs the python implementation of libsvm - train then classify
%
% FYI: format for libsvm is: 
% <label> <index1>:<value1> <index2>:<value2> ... \n
% <label> <index1>:<value1> <index2>:<value2> ... \n
%
%

% the data matrices for controls [C] and FTD [F]:
%load('BParams_For_SVM','C','F');

nlou = 5000; %# number of leave-one-outs

%# check same # vectors
if size(C,2) ~= size(F,2); error('check number of vectors'); end

%# truth vectors
tC = ones(length(C),1)*+1;
tF = ones(length(F),1)*-1;

%# make sure there's both cases in training set by splitting
half1 = 1:round(length(tC)/2);
half2 = half1(end)+1:round(length(tC));

TheData = ...
    [tC(half1) C(half1,:);
     tF(half1) F(half2,:);
     tC(half2) C(half2,:);
     tF(half2) F(half2,:)];

Data = TheData; 

for i = 1:nlou
    
    %# leave-one-out 
    nt  = size(Data,1);        %# full size
    exc = randi(nt);           %# int index to exclude
    a   = 1:nt;                %# full ind vector
    a   = a(~ismember(a,exc)); %# minus the exc: a(~i,:)

    TheData = Data(a,:);
    
    d.Truths{i}  = TheData(1:half1(end)*2-1,1);
    d.TheData{i} = TheData(:,[2:end]);
    
    %# write as csv
    csvwrite('TheData',TheData); 

    %# convert csv to libsvm format using c func
    if ~exist('maker.out');
        !gcc convert.c -o maker.out
    end

    %# save as file 'formd' for passing to python
    [OK,~] = unix(['./maker.out TheData > formd']); % libsvm formatting

    if ~OK ; fprintf('wrote datafile\n'); end

    %# call DoPyTrain [alex] and save result to 'output'
    !cp formd libsvm-3.21/python/
    cd       libsvm-3.21/python/

    cmdstr  = ['./DoPyTrain.py ',num2str(round(size(TheData,1)*.5)),' > output'];
    [ok,ii] = unix(cmdstr);

    if ~ok ; fprintf('success\n'); end

    %# read output file....
    result = fopen('output');
    resdat = textscan(result,'%s');
    fclose(result);

    resdat      = resdat{:};
    d.N_Iter{i} = str2num(resdat{6});
    d.Nu{i}     = str2num(resdat{9});
    d.obj{i}    = str2num(resdat{12});
    d.rho{i}    = str2num(resdat{15});
    d.nSV{i}    = str2num(resdat{18});
    d.nBSV{i}   = str2num(resdat{21});
    d.Accur{i}  = str2num(strrep(resdat{28},'%','')); % percent

    div       = find(resdat{29}=='/');
    d.Corr{i}   = resdat{29}(regexp(resdat{29}(1:div),'\d'));
    d.Tot{i}    = resdat{29}(div+regexp(resdat{29}(div+1:end),'\d'));

    fprintf(['Correct:  ',d.Corr{i}, ' / ', d.Tot{i}, '\n']);
    
    % retrieve predictions [funny formatting]
    for j = 1:round(size(TheData,1)*.5)-1
        	k = strrep(strrep(strrep(resdat{30+j},'[',''),',',''),']','');
            pred(j) = str2num(k);
    end
    
    TvPred = [TheData(41:end,1),pred'];
    d.TruthAndPrediction{i} = TvPred;
    
    d.P(i) = predictive(TvPred*-1+1);
    
    cd ../..

end

d.Accur = cat(2,d.Accur{:});

% histogram with est pdf
A = d.Accur;
u = unique(A);
histfit(A,length(u));
set(gca,'fontsize',45);
xlabel('Accuracy (%)','fontsize',38);
ylabel('frequency [cycles]','fontsize',38);
title('Accuracy over 5000 cycles with leave-one-out','fontsize',38)
xlim([50 100])

% average true / false pos / neg [scaled]
mp.TP = mean(cat(2,d.P.TP));
mp.FP = mean(cat(2,d.P.FP));
mp.TN = mean(cat(2,d.P.TN));
mp.FN = mean(cat(2,d.P.FN));

ss = mp.TP + mp.FP + mp.TN + mp.FN;

mp.TP = mp.TP / ss*100;
mp.FP = mp.FP / ss*100;
mp.TN = mp.TN / ss*100;
mp.FN = mp.FN / ss*100;

mp.PPV  = mean(cat(2,d.P.PPV));
mp.NPV  = mean(cat(2,d.P.NPV));
mp.Sens = mean(cat(2,d.P.Sensitivity));
mp.Spec = mean(cat(2,d.P.Specificity));

d.mp    = mp; % return


%javaaddpath('/Users/Alex/Desktop/MMN/AllSubs/libsvm-3.21/java/')


%cd libsvm-3.21/java/
%[Done,Out] = unix(['java svm_train ../../formd'])
