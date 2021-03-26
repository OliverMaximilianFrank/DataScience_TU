% Data Science for Human Factors - Assignment 15
% Dozent: Marius Klug
% Student: Oliver Frank 
% MATLAB Version: R2020b

% Load Excel Sheet into MATLAB 
cd 'C:\Users\olive\OneDrive\Desktop\Data Science HA\Assignment_15_Oliver_Frank'
data = readtable('Data\Data Science for Human Factors (Antworten)');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% CLEANING 

% Get Class of interesting variable 
class(data.Zeitstempel)
% Seperate it from data table and seperate its values
t = datetime(data.Zeitstempel);
v = datevec(data.Zeitstempel);
% Focus on date not time, and on year not day/month 
Var1 = datetime(v(:,1:3));
Var2 = datevec(Var1);
% delete all rows from all surveys 2019 
for i = 1:height(Var2)
    if Var2(i) == 2019
        data(1,:) = [];
    end 
end 
% 188 Participants left


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% USABLE FORM 

% Delete empty Columns 
delete_cols = [2,5,6,7,12,13,15,18,21,25,28,29,34,36,37,39,42,43,44,45,...
    46,50,52,53,54,56,67,82,83];
fin = width(delete_cols);

for i = 1:fin
    % data(:,delete_cols(i))
    data(:,delete_cols(i)) = [];
    delete_cols = delete_cols - 1;
end 

% Translate answers from 16 Variables from chars to nums/ doubles
% For better operation possibilities change table to cell
data_cell = table2cell(data);

% Loop through all rows and translate the strings into nums 
for f = 1:height(data_cell)
% Every field in that column of the struct gets checked 
D = data_cell(f,23);
    if strcmp(D,'In der Waffel') 
        data_cell{f,23} = 1;
    else
        data_cell{f,23} = 2;
    end 
end 

% We have to dot that with 16 Variables, so that might be function worthy
% Reset data_cell 
data_cell = table2cell(data);


% Before: Set all NaN's and [] to zero 
% Look for every cell's value if it is NaN or empty and if so set it to 0
for d = 2:width(data_cell)
    for s = 1:height(data_cell)
        if isnan(data_cell{s,d}) 
            data_cell{s,d} = 0;
        end 
        if isempty(data_cell{s,d}) 
            data_cell{s,d} = 0;
        end 
    end 
end 

% Small adjustment for Variable 50 and 52 
% set seven extraordinary answer examples to its categories 
data_cell{147,50} = 0;
data_cell{105,50} = 0;
data_cell{47,52} = 'Schüler_In';
data_cell{64,52} = 'selbstständig';
data_cell{103,52} = 'Student_In';
data_cell{186,52} = 'Student_In';
data_cell{188,52} = 'Student_In';


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% USE FUNCTION

% Use function for all Variables that have to be converted 
% Create two arrays for all the variables that have to be converted and
% their corresponding amount of numeric labels 
var_arr = [23 33 34 35 36 37 38 39 40 41 42 43 51 52 53 54];
lbl_arr = [2 2 2 2 2 2 2 2 3 2 2 3 6 5 2 2];
% Put all the conversion information into one meta_cell array 
meta_cell = cell(width(var_arr),1);
% First column for the number of the variables 
for c = 1:width(var_arr)
    meta_cell{c,1} = var_arr(c);
end 
% go through var_arr and use the function for every variable placeholder in
% there and transform its char categories to nums corresponding to the
% associated label number in lbl_arr
for q = 1:width(var_arr)
    
    [data_cell, dicti, label_count] = kat2num(data_cell,var_arr(q),lbl_arr(q));
    % Put in every iteration the information into the meta_cell array for
    % later better understanding when using the data for calculation 
    for n = 1:label_count
        meta_cell{q,2*n} = dicti{n,1};
        meta_cell{q,2*n+1} = dicti{n,2};
    end
    
end 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% VISUALIZATION 
% Main Variable of Interest: Windows or Mac(41)?  
% How is our main variable distributed on the the three variables education
% (51), employment relationship(52) and birthyear(50)?
% Is there maybe a difference on these variables between Mac and Windows
% Users? 

% First change the variable birthyear(50) to a age variable for better
% reading on the plot
for p = 1:height(data_cell)
    if data_cell{p,50} ~= 0
        data_cell{p,50} = 2021 - data_cell{p,50};
    end 
end 
% the so extracted age shouldn't be seen completely valide because the day
% and month of birth was not used. So they have to be seen as an
% approximation with the possibility of -1 

% Plot all three Variables and divide the data by the main variable through
% two different colors

% First change all variable of interest to mat for better operating 
column_tech = cell2mat(data_cell(:,41));
column_edu = cell2mat(data_cell(:,51));
column_emp = cell2mat(data_cell(:,52));
column_age = cell2mat(data_cell(:,50));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% START FIGURE
% Open figure one and clear it, just in case
figure(1); clf
% Set directly white
set(gcf,'color','w');


% FIRST SUBPLOT
subplot(311)

% divide the education variable by its windows/mac class
windows_edu = column_edu(column_tech==1);
mac_edu = column_edu(column_tech==2);

% Count the amount of every education level in each main variable class 
wones_edu = sum(windows_edu(:) == 1);
wtwos_edu = sum(windows_edu(:) == 2);
wthrees_edu = sum(windows_edu(:) == 3);
wfours_edu = sum(windows_edu(:) == 4);
wfives_edu = sum(windows_edu(:) == 5);
wsixs_edu = sum(windows_edu(:) == 6);

% Array to combine the counts
% my kat2num function does not register ordinal data, so I have to change
% the order in a meaningful way (Adjustment of the function will follow 
% with more time ;))
windows_edus = [wfours_edu wtwos_edu wsixs_edu wfives_edu wones_edu wthrees_edu];

% Analoug for mac 
mones_edu = sum(mac_edu(:) == 1);
mtwos_edu = sum(mac_edu(:) == 2);
mthrees_edu = sum(mac_edu(:) == 3);
mfours_edu = sum(mac_edu(:) == 4);
mfives_edu = sum(mac_edu(:) == 5);
msixs_edu = sum(mac_edu(:) == 6);

mac_edus = [mfours_edu mtwos_edu msixs_edu mfives_edu mones_edu mthrees_edu];

% Combine both in one array 
both_edu = [windows_edus; mac_edus];
% Plot it as a bar plot 
color_vector = [100 100 100; 200 200 200];
b1 = bar(1:6,both_edu,'stacked','FaceColor','flat');
b1(1).CData = [0 0 0.5];
b1(2).CData = [1 1 1];
% Make it prettier..
ax1 = gca;
ax1.Color = 'w';
ax1.FontSize = 10;
ax1.FontName = 'Times';
% Create general title 
title({'Demographische Untersuchung von Windows und Mac Usern',''}, 'FontSize', 18)
% Definition of labels 
xlabel('Ausbildung', 'FontSize', 14,'FontWeight', 'bold', 'FontName', 'Times')
ylabel('Absolute Häufigkeit', 'FontSize', 14,'FontWeight', 'bold', 'FontName', 'Times')
xticklabels({'Hauptschule','Realschule','Abitur','Bachelor','Master','Doktor'})
% Legend and its position 
leg1 = legend('Windows User', 'Mac User');
set(leg1, 'Position', [0.20, 0.85, 0.05, 0.05])

% Thoughts:
% It looks like Mac User tend to have a slightly higher education level in
% our sample ... Surprisingly ;) And Windows seems to be the more used
% operation system.


% SECOND SUBPLOT
subplot(312)

% divide the employment relationship variable by its windows/mac class
windows_emp = column_emp(column_tech==1);
mac_emp = column_emp(column_tech==2);

% Same as above...
wones_emp = sum(windows_emp(:) == 1);
wtwos_emp = sum(windows_emp(:) == 2);
wthrees_emp = sum(windows_emp(:) == 3);
wfours_emp = sum(windows_emp(:) == 4);
wfives_emp = sum(windows_emp(:) == 5);

windows_emps = [wfours_emp wtwos_emp wfives_emp wones_emp wthrees_emp];

mones_emp = sum(mac_emp(:) == 1);
mtwos_emp = sum(mac_emp(:) == 2);
mthrees_emp = sum(mac_emp(:) == 3);
mfours_emp = sum(mac_emp(:) == 4);
mfives_emp = sum(mac_emp(:) == 5);

mac_emps = [mfours_emp mfives_emp mthrees_emp mones_emp mtwos_emp];

% Combine both in one array 
both_emp = [windows_emps; mac_emps];
% Plot it as a bar plot 
b2 = bar(1:5,both_emp,'stacked','FaceColor','flat');
b2(1).CData = [0 0 0.5];
b2(2).CData = [1 1 1];
% Make it pretty 
ax2 = gca;
ax2.FontSize = 10;
ax2.FontName = 'Times';
% Definition of labels 
xlabel('Beschäftigung', 'FontSize', 14,'FontWeight', 'bold', 'FontName', 'Times')
ylabel('Absolute Häufigkeit', 'FontSize', 14,'FontWeight', 'bold', 'FontName', 'Times')
xticklabels({'Arbeitslos','Schüler_In','Student_In','Angestellt','Selbstständig'})
% Legend and its position 
leg2 = legend('Windows User', 'Mac User');
set(leg2, 'Position', [0.20, 0.55, 0.05, 0.05])

% Thoughts:
% Seems like Professionals tend to use Microsoft while students prefer Mac, 
% which makes sense in terms of that Mac is mostly famous for its style and 
% easy usability, while there are better options to customize the usage of 
% the Windows operation system which might be better in a professional 
% context where the users have very different requirements on the system.
% Mac migth be a hip thing though and like we all know students are
% hipsters ;)


% THIRD SUBPLOT
subplot(313)
% divide the employment relationship variable by its windows/mac class
windows_age = column_age(column_tech==1);
mac_age = column_age(column_tech==2);
% Clean from zero 
windows_age(windows_age == 0) = [];
mac_age(mac_age == 0) = [];
% Show the distribution of age divided by the main variable on one plot 
h1 = histogram(windows_age,50,'Normalization','pdf');
h1.FaceColor = [0 0 0.5];
hold on 
h2 = histogram(mac_age,75,'Normalization','pdf');
h2.FaceColor = [1 1 1];
% Make it pretty 
ax3 = gca;
ax3.FontSize = 10;
ax3.FontName = 'Times';
% Definition of labels 
xlabel('Alter', 'FontSize', 14,'FontWeight', 'bold', 'FontName', 'Times')
yla = ylabel('Relative Häufigkeit', 'FontSize', 14,'FontWeight', 'bold', 'FontName', 'Times');
yticks([.1 .2])
% Legend and its position 
leg3 = legend('Windows User', 'Mac User');
set(leg3, 'Position', [.20, .25, .05, .05])
% Set x axis limits 
xlim([0 100])

% Add a Density functions to look for the normal distributions
% Windows
% Find the rigth mu and sigma
pd1 = fitdist(windows_age,'Normal'); 
% Plot the density function
y = 0:1:100;
mu1 = pd1.mu;
sigma1 = pd1.sigma;
f = exp(-(y-mu1).^2./(2*sigma1^2))./(sigma1*sqrt(2*pi));
win_c = [0 0 0.5];
pl_win = plot(y,f,'LineWidth',1.5,'Color',win_c);
% Mac
% Find the rigth mu and sigma
pd2 = fitdist(mac_age,'Normal'); 
% Plot the density function
y = 0:1:100;
mu2 = pd2.mu;
sigma2 = pd2.sigma;
f = exp(-(y-mu2).^2./(2*sigma2^2))./(sigma2*sqrt(2*pi));
mac_c = [0.4 0.4 0.4];
pl_mac = plot(y,f,'LineWidth',1.5,'Color',mac_c);
% Name the two normal densitiy functions 
leg3.String{3} = 'Windows NV';
leg3.String{4} = 'Mac NV';

% Thoughts:
% One can see that both groups are gathering equally around the mean of 27
% more or less. Both Histogramms look more or less normal distributed, the
% two density functions show how the ideal distributions should look like.
% One can see that they are different, so they can not obviously be
% approved. Nevertheless we will calculate a small ttest to check if there 
% is a significant difference between the two groups on age, which 
% one would probably not assume just by looking on the plot. 


% two-sample t-test 
% Aim: Is there a significant difference between Mac Users and Windows
% Users in terms of age? (I assume equal variance and normal distribution)

result = ttest2(windows_age, mac_age);
% result = 0 -> there is no significant difference between the the two
% groups on the variable age in our sample which we take as the best 
% representation for the basic population behind it, theoretically covered 
% by our assumptions...
% So, we could prove statistically (on a very basic and scientifically not 
% sufficient way ;)) what we have already seen and prognosed by our plot.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% MACHINE LEARNING 


% TSNE

% Create a Matrix out of the preused variables 
X = [column_edu column_emp column_age column_tech];
% find all orws and cols in X with value 0
[a b] = find(X==0);
a = sort(a);
% delete all rows wirh at least one value of 0
for i=1:length(a)
    X(a(i),:) = [];
    a = a-1;
end 

X_clean = X;
% Create two other matrices grouped again by windows/mac
X_win = [X_clean(X_clean(:,4) == 1,1) X_clean(X_clean(:,4) == 1,2) X_clean(X_clean(:,4) == 1,3)];
X_mac = [X_clean(X_clean(:,4) == 2,1) X_clean(X_clean(:,4) == 2,2) X_clean(X_clean(:,4) == 2,3)];

figure(2); clf
set(gcf,'color','w');
% Plot normal
subplot(231)
scatter3(X(:,1),X(:,2),X(:,3))
title('3D Data')
view(-30,10)
set(gca,'FontName','Times')
xlim([0 7])
ylim([0 7])
zlim([0 100])
xlabel('Education')
ylabel('Employment')
zlabel('Age')

% Plot windows normal
subplot(232)
scatter3(X_win(:,1),X_win(:,2),X_win(:,3))
title('3D Data Windows')
view(-30,10)
set(gca,'FontName','Times')
xlim([0 7])
ylim([0 7])
zlim([0 100])
xlabel('Education')
ylabel('Employment')
zlabel('Age')

% Plot mac normal 
subplot(233)
scatter3(X_mac(:,1),X_mac(:,2),X_mac(:,3))
title('3D Data Mac')
view(-30,10)
set(gca,'FontName','Times')
xlim([0 7])
ylim([0 7])
zlim([0 100])
xlabel('Education')
ylabel('Employment')
zlabel('Age')


% Plot normal with tsne reduction 
subplot(234)
Y = tsne(X,[],2,3);
scatter(Y(:,1),Y(:,2))
title({'Dimensionality Reduction', 'of Data with t-SNE'})
xlabel('t-SNE 1')
ylabel('t-SNE 2')
set(gca,'FontName','Times')

% Plot windows with tsne reduction 
subplot(235)
Y = tsne(X_win,[],2,3);
scatter(Y(:,1),Y(:,2))
title({'Dimensionality Reduction', 'of Windows Data with t-SNE'})
xlabel('t-SNE 1')
ylabel('t-SNE 2')
set(gca,'FontName','Times')

% Plot mac with tsne reduction
subplot(236)
Y = tsne(X_mac,[],2,3);
scatter(Y(:,1),Y(:,2))
title({'Dimensionality Reduction', 'of Mac Data with t-SNE'})
xlabel('t-SNE 1')
ylabel('t-SNE 2')
set(gca,'FontName','Times')


% Font Size for all
set(findall(gcf,'-property','FontSize'),'FontSize',12)
% Global title 
a = axes;
glo_t = title('Visualization of the data with and without tsne dimension reduction'...
    ,'Position', [0.5, 1.05],'FontSize',15,'FontName','Times');
a.Visible = 'off'; 
glo_t.Visible = 'on'; 


% Thoughts:
% It is very nice that just by chance in every tsne-dimension-reduction was
% reduced a different variable. On the 3D Data it was employment, on 3D
% Data Windows it was education and in 3D Data Mac it was age. From my
% point of view that has worked out pretty well for the first two
% reductions and still contain main parts of the information of the reduced 
% variable, while on the third plot the information of the age is hard to 
% find. The reason might be the difference between the variable types of 
% metric and categorical. 


% CLUSTERING

% Preword: I don't think that Clustering is a very good idea in that
% dataset, because we only have categorical data, except for age.
% Clustering takes advantage of the distance between metric variables to
% group them. Categorical data, especially in the cases where the order/
% number of the category doesn't relate to its strength, can't give that
% information completely or not as good as it should to create reliable
% clusters. Anyway as I understood the task I will still try to dot it at 
% least once and discuss the weaknesses on the plot afterwards. 


% kmeans 
k = 3; 
% Remove the Outlier
X(17,:) = [];

[groupidx,cents,sumdist,distances] = kmeans(X,k);

% cluster colors
lineColors = [0,0.75,0.75;0.75,0,0.75;0.75,0.75,0];

figure(3);clf;
set(gcf,'color','w');

hold on
% now draw the raw data in different colors
for i=1:k
    % data points
    scatter3(X(groupidx==i,1),X(groupidx==i,2),X(groupidx==i,3),'MarkerFaceColor',lineColors(i,:))
    % centroids 
    scatter3(cents(i,1),cents(i,2),cents(i,3),'MarkerFaceColor',[0 0 0]);
end

% connecting lines for clarification
for i=1:length(X)
   
    plot3([X(i,1) cents(groupidx(i),1)],...
        [X(i,2) cents(groupidx(i),2)],...
        [X(i,3) cents(groupidx(i),3)], 'color', ...
        lineColors(groupidx(i),:))
end

% Make it pretty 
title({'Clustering on one metric and two categorical variables (k = 3)'},...
    'FontSize',15,'FontName','Times')
xlim([0 7])
ylim([0 7])
zlim([0 100])
view(154,12)
xlabel('Education')
ylabel('Employment')
zlabel('Age')
set(gca,'FontName','Times')
grid on 


% Thoughts: As we can see the function of course is still working and the
% clustering on the age axis, which is the only metric variable, seems to
% fit preety well and plausible. But as we rotate, so that we can only see
% the x and z axis, we directly kow that the clustering can't make any
% meaningful sense. If, like I already mentioned, there were sth like an
% ordered likert scale, it would still be to some extent justifiable to use 
% this algo to cluster the data, but in our case it doesn't. We just go on.


% CLASSIFICATION
% Predict Choice of Technology (Mac or Windows) by education and age 

% Clean from outlier didn't help to improve the LDA prediction 
% X_clean(17,:) = [];

% Y (windows/mac) should be predicted by X (age and employment relationship)
% The X_clean is brought from above (362-372)
X = [X_clean(:,2) X_clean(:,3)];
Y = X_clean(:,4);

% compute linear discriminant coefficients
W = LDA(X,Y);
% Predict Classes 
Predictions = LDA_predict(X,W);

% plot
figure(4);clf

set(gcf,'color','w');
subplot(2,2,1)
scatter(X(:,1),X(:,2),[],Y,'filled')
title('Actual Classes')
xlabel('Employment')
ylabel('Age')
xlim([0 7])

subplot(2,2,2)
scatter(X(:,1),X(:,2),[],Predictions,'filled');
title('LDA Predicted Classes')
xlabel('Employment')
ylabel('Age')
xlim([0 7])


% As we can see the Clustering with the categorical, not even likert like 
% scale, again is a problem, that doesn't surprise very much. 
% So we will try it out with a categorical variable, where the steps have 
% ordinal information. 


% Predict Choice of Technology (Mac or Windows) by programming and age

% Programming is cool -> new variable to give it a try
column_code = cell2mat(data_cell(:,32));

X = [column_code column_age column_tech];
% find all orws and cols in X with value 0
[a b] = find(X==0);
a = sort(a);
% delete all rows wirh at least one value of 0
for i=1:length(a)
    X(a(i),:) = [];
    a = a-1;
end 

X_clean = X;

% Y (windows/mac) should be predicted by X (age and employment relationship)
X = [X_clean(:,1) X_clean(:,2)];
Y = X_clean(:,3);
size(X)
size(Y)
% compute linear discriminant coefficients
W = LDA(X,Y);
% Predict Classes 
Predictions = LDA_predict(X,W);

% plot
subplot(2,2,3)
scatter(X(:,1),X(:,2),[],Y,'filled')
title('Actual Classes')
xlabel('Code Enthusiast')
ylabel('Age')

subplot(2,2,4)
scatter(X(:,1),X(:,2),[],Predictions,'filled')
title('LDA Predicted Classes')
xlabel('Code Enthusiast')
ylabel('Age')

% Make it prettier 
set(findall(gcf,'-property','FontSize'),'FontSize',12)
% Global title 
a = axes;
glo_t = title('LDA Predictions with different two different variable combbinations'...
    ,'Position', [0.5, 1.05],'FontSize',13,'FontName','Times');
a.Visible = 'off'; 
glo_t.Visible = 'on'; 


% Thoughts: Similiar situation. At least we have one Data Point that is
% classified different to group 1 and also correctly to Mac. Something is
% going wrong here and I guess again it is just the nature of the
% categorical variable or the lack of correlation of the variables or even 
% the model assumptions. With more time to spnd on that, I guess I would
% find the problem and better possibilities to predict my dichotom
% variable.


% Examining if classification exceeds statistical chance of alpha = .5 for
% the second predicition with 'programming' and 'age' as predictive
% variables

[accuracy, confusion_matrix] = compute_classification_accuracy(Y,Predictions);
% accuracy = 0.69

% find the amount of data points in each actual class
classes_n = sum(confusion_matrix);

% Significance Level 
alpha_level = 0.05;

% Test 
pconf = simulateChance(classes_n,alpha_level);
% Significance reached with 0.63 accuracy or 113 correct predicted examples


% Thoughts: We have a significant result, so our predictive model is
% significantly predicting more correct examples than wrong on an alpha
% significance level of 0.05. Nevertheless the result shouldn't make me too
% happy. The model predicted almost every exampe as group 1, this group is
% just over represented, so the chances were high to exceed the 0.05 level.
% Like I mentioned the model should definitely not be used to predict so
% far. Further examinations would improve it for sure...


