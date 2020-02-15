number_of_datasets = 3;

for current_dataset = 1:number_of_datasets
    
    if(current_dataset == 1)
        
        % load in proper workspace (comment out if creating new splits)
        load('spam_workspace.mat');
        
        % spam initialization
        load spam.txt
        W=spam;
        s=size(W); % rows x columns
        m=s(2); % number of columns
        X=W(:,1:m-1); % removes last column
        y=W(:,m); % is only last column
        
    elseif (current_dataset == 2)
        
        % load in proper workspace (comment out if creating new splits)
        load('SAheart_workspace.mat');
        
        % SAheart initialization
        W = readtable("SAheart.txt");
        s=size(W);
        n=s(1); % number of rows
        replace_famhist=zeros(n, 1); % initializes replacement vector
        
        for i = 1:n % fills replacement vector
            if(strcmp(W.famhist(i), 'Present'))
                replace_famhist(i) = 1;
            else
                replace_famhist(i) = 0;
            end
        end
        
        W.famhist = replace_famhist;
        Z = table2array(W);
        
        s = size(Z); % number of total columns
        m = s(2); % -1 for for deleted first col
        
        X=Z(:, 2:m-1); % removes first label column and last indicator column
        y=Z(:,m); % is only last column
        m = m-1;

    elseif (current_dataset == 3)
        
        % load in proper workspace (comment out if creating new splits)
        load('zip_workspace.mat');

        % % zip train initialization
        load zip.train
        W=zip;
        s=size(W); % rows x columns
        m=s(2); % number of columns

        del_rows_0s = find(W(:,1)==0);
        del_rows_1s = find(W(:,1)==1);

        Wc= cat(1, del_rows_0s, del_rows_1s);

        new_W=W(Wc,:);
        % s=size(W); % rows x columns
        % n=s(1); % number of rows

        X=new_W(:,2:m); % removes first column
        y=new_W(:,1); % is only first column

        s = size(X);
        n=s(1);
        
    end % ends all initializations
       


    % Xv = validate
    % Xt = training
    % Xte = testing

     Xs=X; % scaled input set
     indexes_to_remove = [];

     for i=1:m-1 % loops through columns
         u=mean(X(:,i));
         if(std(X(:,i)) == 0)
             Xs(:,i) = zeros(n,1);
         else
        v=std(X(:,i));
        Xs(:,i)=(Xs(:,i)-u)/v;
         end
     end
     Xs;
    %  if(indexes_to_remove ~= [])
    %      for i = 1:size(indexes_to_remove)
    %          X(
    %        


    % scaled labels
    ys=y; 
    ys(y==0)=-1;
    ys;

    %[X1,y1,Xv,yv]=splitdata(Xs, ys, 0.6); % splits into 60/40 ratio
    %[X2,y2,X3,y3]=splitdata(Xv, yv, 0.5); % finishes splitting into 60/20/20

    n=1500; % number of iterations
    weightMatrix=GradientDescent(X1,y1,0.1,n);
    s1=size(X1); % train
    s2=size(X2); % validate
    s3=size(X3); % test

    %mean logistic loss
    sum=0;
    sum1=0;

    for j=1:s1(1)
     l=(1/s1(1))*log(1+exp(-y1(j)*weightMatrix'*X1(j,:)'));
     sum=sum+l;
    end

    for j=1:s2(1)
     l1=(1/s2(1))*log(1+exp(-y2(j)*weightMatrix'*X2(j,:)'));
     sum1=sum1+l1;
    end

    % initializing vectors
    w1=zeros(s1(1),n); % to be training weight vector
    w2=zeros(s2(1),n); % to be validate weight vector
    w3=zeros(1,n); % to be test weight vector

    % for k=1:s1(1)
    % w1(:,k)=weightMatrix'*X1(k,:)';
    % end
    % for k=1:s2(1)
    % w2(:,k)=weightMatrix'*X2(k,:)';
    % end

    % assignign results from training set
    p1=zeros(1,n);
    w=(weightMatrix(:,1:n)'*X1(1:s1(1),:)')';
    w1(w<=0)=-1;
    w1(w>0)=1;

    % counts the number of correct predictions
    for l=1:n
    x=w1(:,l);
    p=size(x(x==y1));
    p1(l)=p(1);
    end

    % assinging results from validation set
    p2=zeros(1,n);
    w=(weightMatrix(:,1:n)'*X2(1:s2(1),:)')';
    w2(w<=0)=-1;
    w2(w>0)=1;

    % counts the number of correct predictions
    for l=1:n
        x=w2(:,l);
        p=size(x(x==y2)); 
        p2(l)=p(1);
    end

    % assinging results from test set

    w=(weightMatrix(:,1:n)'*X2(1:s2(1),:)')';
    w2(w<=0)=-1;
    w2(w>0)=1;

    for l=1:n
    x=w2(:,l);
    p=size(x(x==y2));
    p2(l)=p(1);
    end

    % calculating error percent
    pe1=((1-p1/s1(1))*100)';
    pe2=((1-p2/s2(1))*100)';

    %find optimal
    % console display
    [a,b]=min(sum1);
    b; % optimum iterations

    [a2,b2]=min(pe2);
    b; % optimum iterations

    [a1,b1]=min(pe1);
    b; % optimum iterations

    % log_loss_graph_labels = [b, n];

    % matlab plotting

    % plotting mean log loss
    figure(current_dataset);
    hold on;
    plot(sum','g-');
    plot(sum1','r-');
    plot(b, a, 'o');
    %plot(n, a2, 'o'); % need to find y value of n
    % graph display utility
    title("Dataset: " + current_dataset + ", Logistic Loss");
    xlabel('Iterations');
    ylabel('Mean Log Loss');
    legend({'training', 'validation', 'min'});
    ax = gca;
    ax.FontSize = 14;
    hold off;

    % plotting Error Percent
    figure(current_dataset + 100); % breaks if more than 100 datasets
    hold on;
    plot(pe1,'g-');
    plot(pe2,'r-');
    % point labelling
    plot(b1, a1, 'o');
    plot(b2, a2, 'o');
    % graph display utility
    title("Dataset: " + current_dataset + ", Error Rate");
    xlabel('Iterations');
    ylabel('% Error');
    legend({'training', 'validation', 'min', 'min'});
    ax = gca;
    ax.FontSize = 14;
    hold off;


    % display test values
    % counts the number in each class (0 or 1)
    % use these numbers for paper report
    count_yt_0s = size(y1(y1==-1));
    count_yv_0s = size(y2(y2==-1));
    count_yte_0s = size(y3(y3==-1));

    count_yt_1s = size(y1(y1==1));
    count_yv_1s = size(y2(y2==1));
    count_yte_1s = size(y3(y3==1));

    y_hat=(weightMatrix(:,b)'*X3(1:s3(1),:)')'; % for plotting in R

    p3=zeros(s3(1),1);
    p3(y_hat<=0)=-1;
    p3(y_hat>=0)=1;


    x=p3;
    p=size(x(x==y3));
    e=p(1);


    % calculating error percent

    pe3=((1-e/s3(1))*100)
    
end % ends data set loop
