% Pere Marti-Puig 31/05/2022




clear all; close all; clc;

% Code, images and program directories
dir_base=pwd;                               
DIR_IMATGES=[dir_base(1:end-1) 'im'];
DIR_TEXT=[dir_base(1:end-1) 'txt'];


BCN04=dir([DIR_IMATGES '/*pvt04.*']);  % Timex
fields = {'folder','date','bytes','isdir','datenum'};
BCN04 = rmfield(BCN04,fields);


FR04=dir([DIR_TEXT '/*04FR*.txt']);
GS04=dir([DIR_TEXT '/*04GS*.txt']);
JA04=dir([DIR_TEXT '/*04JA*.txt']);

Num_04=numel(BCN04);

    
%% Collect marker points, splines and continuous shorelines of all images in 
% the BCN04 struct, then store it in BCN04.mat
% If BCN04.mat already exists, we read it


if isfile('BCN04.mat')
     % File exists.
     load BCN04.mat
else
    % File does not exist.
    PINI=40; % Starting pixel of splines
    
    for i=1:Num_04
       
        % Reading of the points of each marker
        xy_FR=import_TXT_file([DIR_TEXT '/' FR04(i).name]); 
        xy_GS=import_TXT_file([DIR_TEXT '/' GS04(i).name]); 
        xy_JA=import_TXT_file([DIR_TEXT '/' JA04(i).name]); 
       
        % Splines of the markers 
        [xs_FR, ys_FR]=splineCalc(xy_FR); % Spline FR
        idx=find(xs_FR<PINI); xs_FR(idx)=[]; ys_FR(idx)=[];
        idx=find(xy_FR(:,1)<PINI); xy_FR(idx,:)=[];

        [xs_GS, ys_GS]=splineCalc(xy_GS); % Spline GS
        idx=find(xs_GS<PINI); xs_GS(idx)=[]; ys_GS(idx)=[];
        idx=find(xy_GS(:,1)<PINI); xy_GS(idx,:)=[];

        [xs_JA, ys_JA]=splineCalc(xy_JA); % Spline FR
        idx=find(xs_JA<PINI); xs_JA(idx)=[]; ys_JA(idx)=[];
        idx=find(xy_JA(:,1)<PINI); xy_JA(idx,:)=[];

        % The TRUE coastline will be the average of the individual splines
        xC_Fi=max([max(xs_GS) max(xs_FR) max(xs_JA)]);

        xC_=PINI:1:xC_Fi;
        yC_=zeros(size(xC_));

        n_pixels=length(xC_);

        for ii=1:n_pixels
            N=0; y=0;
            i_GS=find(xs_GS==xC_(ii));
            i_FR=find(xs_FR==xC_(ii));
            i_JA=find(xs_JA==xC_(ii));

            if ~isempty(i_GS) y=y+ys_GS(i_GS); N=N+1; end
            if ~isempty(i_FR) y=y+ys_FR(i_FR); N=N+1; end
            if ~isempty(i_JA) y=y+ys_JA(i_JA); N=N+1; end

            yC_(ii)=y/N;
        end
        
        % marker points
        BCN04(i).xy_FR=xy_FR;
        BCN04(i).xy_GS=xy_GS;
        BCN04(i).xy_JA=xy_JA;
        
        % individual splines
        BCN04(i).s_FR=[xs_FR' ys_FR'];
        BCN04(i).s_GS=[xs_GS' ys_GS'];
        BCN04(i).s_JA=[xs_JA' ys_JA'];

        % true shoreline
        BCN04(i).xy=[xC_' yC_'];

    end
    save('BCN04.mat','BCN04');

end


%% Control plot to visualize information

% View PantViewTimex images  if View=1
% Add experts' marks if Points=1
% Add individual splines if Splines=1
% Add continous shoreline if Shoreline=1

% Marker 1: yellow
% Marker 2: red
% Marker 3: green

% Continuous shoreline: black

View=1; % View=0;
Points=1;
Splines=1;
Shoreline=1;

if View==1
    for i=1:Num_04
        I= imread([DIR_IMATGES '/' BCN04(i).name]);       % PantViewTimex #i
        
        figure(i);
        imshow(I); hold on;
        
        if Points==1
            % Inserting the points of each marker
            plot(BCN04(i).xy_FR(:,1),BCN04(i).xy_FR(:,2),'.y')
            plot(BCN04(i).xy_GS(:,1),BCN04(i).xy_GS(:,2),'.r')
            plot(BCN04(i).xy_JA(:,1),BCN04(i).xy_JA(:,2),'.g')
        end

        % Drawing splines for each marker
        if Splines==1
            plot(BCN04(i).s_FR(:,1),BCN04(i).s_FR(:,2),'y')
            plot(BCN04(i).s_GS(:,1),BCN04(i).s_GS(:,2),'r')
            plot(BCN04(i).s_JA(:,1),BCN04(i).s_JA(:,2),'g')
        end
        
        if Shoreline==1
            plot(BCN04(i).xy(:,1),BCN04(i).xy(:,2),'k')
        end
        pause(0.1)
    end
end






%% Split Images randomly into Training and Test groups if IsRandom='yes'. NOTE in this case, we use one test image for Validation
% elsewher take a previously saved partition

IsRandom='yes';


if IsRandom=='yes'

    % Index permutation
    idx_perm=randperm(Num_04);

    idx_train=idx_perm(1:floor(Num_04/2));
    idx_test=idx_perm(floor(Num_04/2)+1:end);
    idx_val=idx_perm(end); % The last test image is used also to validate
    
    % PVTimex
    BCN04_train=BCN04(idx_train); % Information of Train group in  BCN04_train
    BCN04_test=BCN04(idx_test);     % Information of Test group in  BCN04_test
    BCN04_val=BCN04(idx_val);       % Information of Validation group in  BCN04_val

  
    % Save random partition info. Take care not overlap info !!!
    save('DataSplit_0_BCN04.mat','BCN04_train','BCN04_test','BCN04_val','idx_train','idx_test','idx_val');
else
     % Load a particular random partition. 
    load DataSplit_0_BCN04.mat 
end





%% Changing the images from RGB format to LAB, decomposes the set of images into columns, and prepares the target for each column.

% Prepare data for training
[Xlab Ylab ee_train] = Undo_into_Columns_RGB2LAB(BCN04_train,DIR_IMATGES);
% Prepare data for Test
[XlabTest YlabTest ee_test] = Undo_into_Columns_RGB2LAB(BCN04_test,DIR_IMATGES);
%  Prepare data for  Validation
[XlabVal YlabVal ee_val] = Undo_into_Columns_RGB2LAB(BCN04_val,DIR_IMATGES);


%% TRAINING, TEST and Validation
% Summary of available data in form of data and targets

Xlab=Xlab'; 
Ylab=Ylab'; 

XlabTest=XlabTest'; 
YlabTest=YlabTest';

XlabVal=XlabVal'; 
YlabVal=YlabVal';

%% Network configuration
numFeatures=3;          
numHiddenUnits=45;    
numClasses=2;
miniBatchSize=80;

layers=[       sequenceInputLayer(numFeatures)
               bilstmLayer(numHiddenUnits,'OutputMode','sequence')
               fullyConnectedLayer(numClasses)
               softmaxLayer
               classificationLayer ];
           
%% Training parameters
options_Lab= trainingOptions( 'adam',...
                'ExecutionEnvironment','cpu', ...
                'MaxEpochs',3,... 
                'GradientThreshold',1,... 
                'MiniBatchSize', miniBatchSize, ...  
                'Verbose',1,...
                'VerboseFrequency',100,...
                'Plots','none',... %'training-progress',...
                'ValidationData',{XlabVal,YlabVal});


%% Training a new network and SAVE it (new_Train_Lab) if new_Train_Lab='yes' otherwise load a trained one

new_Train_Lab='yes';
%new_Train_Lab='no';

if strcmp(new_Train_Lab,'yes')
    tic;  net_BCN04_Lab=trainNetwork(Xlab,Ylab,layers,options_Lab); toc
        
    c=clock;
    nom2=['Net_BCN04_Lab' '_' num2str(c(1)) '_' num2str(c(2)) '_' num2str(c(3)) '_' num2str(c(4)) '.mat' ];
    

    save(nom2,'net_BCN04_Lab','layers','options_Lab','numHiddenUnits','numFeatures','numClasses')
     
else
    % Load a previous trained network
    load Net_BCN04_Lab_1.mat    % 45 nodes
end







%% image-by-image TEST showing the results

 close all;

 Num_Im_Test =numel(BCN04_test);

 
 ErrorInfoTest=struct('test_im',[],'xy',[],'xy_lab',[]); % To store 'xy' the true line and the stimated line in 'xy_lab'

 
 for jj=1:Num_Im_Test 
     
     % Find the columns of a single test image (ii)
     I=imread( [DIR_IMATGES '/' BCN04_test(jj).name]) ;
     [XlabT YlabT eel] = Undo_into_Columns_RGB2LAB(BCN04_test(jj),DIR_IMATGES);

     % Test for the single image
     YPred_Lab=classify(net_BCN04_Lab,XlabT, 'MiniBatchSize',miniBatchSize, 'SequenceLength','longest');


     figure(jj)

     imshow(I)
     title(['Test on image #' num2str(jj)]);
     x=BCN04_test(jj).xy(:,1); % component x of the TRUE shoreline 
     y=BCN04_test(jj).xy(:,2); % component y of the TRUE shoreline 
     Co=numel(x); % Number of columns
     hold on
    
     % Finding the shoreline point in a column, Column-by-column
     y_lab=[];
     x_lab=[];
     
     for i=1:Co-1 

         % Consider that the shoreline pixel is the firt pixel of water
%          y_pred_Lab=find(YPred_Lab{i}'=='mar');
%          y_pred_Lab=y_pred_Lab(1);   

         y_pred_Lab=find(YPred_Lab{i}'=='terra');
         y_pred_Lab=y_pred_Lab(end);   



         plot(x(i),y_pred_Lab,'.g')

         x_lab=[x_lab x(i)];
         y_lab=[y_lab y_pred_Lab]; 
     end

     pause(0.08)
     plot(x,y,'r')
     pause(0.07)
     hold off
     
     ErrorInfoTest(jj).test_im=['Test' num2str(jj)];
     ErrorInfoTest(jj).xy=[x y];
     ErrorInfoTest(jj).xy_lab=[x_lab' y_lab'];

 end



%%  TRAIN & showing the results image-by-image

ErrorInfoTrain=struct('test_im',[],'xy',[],'xy_lab',[]);

%%
Num_Im_Train =numel(BCN04_train);
 
 for jj=1:Num_Im_Train


     % Find the columns of a single train image (ii)
     I=imread( [DIR_IMATGES '/' BCN04_train(jj).name]) ;
     [XlabT YlabT eel] = Undo_into_Columns_RGB2LAB(BCN04_train(jj),DIR_IMATGES);

     % Test for the single image
     YPred_Lab=classify(net_BCN04_Lab,XlabT, 'MiniBatchSize',miniBatchSize, 'SequenceLength','longest');
   
     figure(jj)
     figure;
     imshow(I)
     title(['Results on Train #' num2str(jj)]);

     x=BCN04_train(jj).xy(:,1); % component x of the TRUE shoreline 
     y=BCN04_train(jj).xy(:,2); % component y of the TRUE shoreline 
     Co=numel(x);
     hold on
    
     y_lab=[];
     x_lab=[];
     for i=1:Co-1

         y_pred_Lab=find(YPred_Lab{i}'=='terra');
         y_pred_Lab=y_pred_Lab(end);    

         plot(x(i),y_pred_Lab,'.g')

         x_lab=[x_lab x(i)];
         y_lab=[y_lab y_pred_Lab]; 
     end

     pause(0.08)
     plot(x,y,'r')
     pause(0.07)
     hold off
     
     ErrorInfoTrain(jj).test_im=['Train' num2str(jj)];
     ErrorInfoTrain(jj).xy=[x y];
     ErrorInfoTrain(jj).xy_lab=[x_lab' y_lab'];

 end


%% Veure train i test

L=numel(ErrorInfoTrain);
E_Train=[];
for i=1:L
    a=ErrorInfoTrain(i).xy(:,2);
    a=a(1:end-1);
    b=ErrorInfoTrain(i).xy_lab(:,2);
    E_Train=[E_Train; mean(abs(a-b))];
    
end

disp('TEST')
L=numel(ErrorInfoTest);
E_Test=[];
for i=1:L
    L
    a=ErrorInfoTest(i).xy(:,2);
    a=a(1:end-1);
    b=ErrorInfoTest(i).xy_lab(:,2);
    E_Test=[E_Test; mean(abs(a-b))];

end



E_Train
mean(E_Train)

E_Test
mean(E_Test)


for i=1:6
    N_Tr{i}=BCN04_train(i).name;
    N_Te{i}=BCN04_test(i).name;
end

Summari_Test=table(N_Te',E_Test,'variablenames',{'Images in the test','Pixel error (avg)'})
AVG_pixel_error_Test=mean(E_Test)
Summari_Train=table(N_Tr',E_Train,'variablenames',{'Images in the training','Pixel error (avg)'})
AVG_pixel_error_Train=mean(E_Train)





%% Functions


function [X Y ee] = Undo_into_Columns_RGB2LAB(Str_info,DIR_IM)

% Description
%   It changes the images from RGB format to LAB, decomposes the set of images into columns, 
%   and prepares the target for each column.
% Inputs
%   Str_info: Structure with the information of the data to be processed
%   DIR_IM: directory of images
% Outputs
%   X: Cell array containing the collumns of all images reported in Str_info in LAB format
%   Y: Cell array containing the categorical target information for all columns of X. Here 0
%   means land and -1 means sea. 
%   ee: total number of columns in X and Y



Num_Im_Train=numel(Str_info);

% Initializing output variables
    ee=0; 
    X={};  
    Y={};  


    for jj=1:Num_Im_Train  % Number of train images
        
        I= imread([DIR_IM '/' Str_info(jj).name]);
        I=rgb2lab(I);

        [R C ~]=size(I);

        x=Str_info(jj).xy(:,1);
        y=Str_info(jj).xy(:,2);
        
        % Control to ensure that the splines do not leave the frame of the images. 
        % NOTE: A particular control is required for each beach 
        
        idx =x>C;
        x(idx)=[]; y(idx)=[];
    
        idx =x<1;
        x(idx)=[];  y(idx)=[];
    
        idx =y<1;
        x(idx)=[]; y(idx)=[];
         
        Co =numel(x); % number of taget columns in image jj
    
        for i=1:Co % 
            ee=ee+1;
            s=zeros(R,3); %  Memory for a column in LAB format.
            c=zeros(R,1); %  Memory for a target
    
            
            % Column
            s(:,1)=I(:,x(i),1); % Component L
            s(:,2)=I(:,x(i),2); % a component
            s(:,3)=I(:,x(i),3); % b component
            
    
            % target
            c(round(y(i))+1:end)=-1; % Mark all pixels below the coastline as sea (-1)
            c=categorical(c, [0 -1], {'terra'   'mar'});
            
            X{ee}=s';    % Incorporation of column s as a row of X
            Y{ee}=c';    % Incorporation of target c as a row of Y
            
        end          
    
    end 

end


function [X Y ee] = Undo_into_Columns__RGB(Str_info,DIR_IM)

% Description
%   Decomposes the set of images into columns maintaining the RGB format, 
%   and prepares the target for each column.
% Inputs
%   Str_info: Structure with the information of the data to be processed
%   DIR_IM: directory of images
% Outputs
%   X: Cell array containing the collumns of all images reported in Str_info in LAB format
%   Y: Cell array containing the categorical target information for all columns of X. Here 0
%   means land and -1 means sea. 
%   ee: total number of columns in X and Y



Num_Im_Train=numel(Str_info);

% Initializing output variables
    ee=0; 
    X={};  
    Y={};  


    for jj=1:Num_Im_Train  % Number of train images
        
        I= imread([DIR_IM '/' Str_info(jj).name]);

        [R C ~]=size(I);

        x=Str_info(jj).xy(:,1);
        y=Str_info(jj).xy(:,2);
        
        % Control to ensure that the splines do not leave the frame of the images. 
        % NOTE: A particular control is required for each beach 
        
        idx =x>C;
        x(idx)=[]; y(idx)=[];
    
        idx =x<1;
        x(idx)=[];  y(idx)=[];
    
        idx =y<1;
        x(idx)=[]; y(idx)=[];
         
        Co =numel(x); % number of taget columns in image jj
    
        for i=1:Co % 
            ee=ee+1;
            s=zeros(R,3); %  Memory for a column in LAB format.
            c=zeros(R,1); %  Memory for a target
    
            
            % Column
            s(:,1)=I(:,x(i),1); % Component L
            s(:,2)=I(:,x(i),2); % a component
            s(:,3)=I(:,x(i),3); % b component
            
    
            % target
            c(round(y(i))+1:end)=-1; % Mark all pixels below the coastline as sea (-1)
            c=categorical(c, [0 -1], {'terra'   'mar'});
            
            X{ee}=s';    % Incorporation of column s as a row of X
            Y{ee}=c';    % Incorporation of target c as a row of Y
            
        end          
    
    end 


end



function xy = import_TXT_file(filename)

    [x y] = textread(filename,'%f %f');
    xy=[x y];

end

function [xs_AA ys_AA]=splineCalc(xy_AA)

% Description
%   splineCalc interpolates the points (x,y) of marker AA from ceil(min(x)) to floor(max(x))
%   in order to all columns between theses two values of the corresponding image have the soreline 
%   point defined.

% Inputs
%   xy_AA: the points (x,y) of marker AA
% Outputs
%   xs_AA: x coordinates after spline interpolation
%   ys_AA: y coordinates after spline interpolation

    x=xy_AA(:,1);  y=xy_AA(:,2);

    [x, ind_unics] = unique(x);
    y=y(ind_unics);

    x_min=ceil(min(x));
    x_max=floor(max(x));

  
  
    xs_AA=x_min:1:x_max;
    ys_AA=round(spline(x,y,xs_AA));

end

