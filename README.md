# Shorelines-biLSTM
shorelines_demo.m 

This demo works with the following directory organization: 

m which contains:

	 shorelines_demo.m (main program) 
	 Net_BCN04_lab_1.mat an already trained bi-LSTM network. 
   
Im which contains a set of 12 images, 6 of which, randomly chosen, will be used to train the network while the remaining 6 will be used for the test.

txt which contains 24 text files with the points entered by each marker. 

Tasks performed by shorelines_demo.m

#1 The construction of a BCN04 structure where it collects the information associated with each image in the following fields: 

	name: name of the image 
	xy_FR: points introduced by marker #1 
	xy_GS: points entered by marker #2 
	xy_JA: points entered by marker #3 
               s_FR: interpolation made from points on scoreboard #1 
               s_GS: interpolation made from points in marker #2 
	s_JA: interpolation made from points in marker #3 
	xy: Continuous coastline that we use to train and test the proposed method 

This structure is built the first time that shorelines_demo.m is executed. Then it is saved in the m directory with the name BCN04, so it doesn’t have to be build it every time. 

#2 The visualization of the available information on each image. That is: the points of each marker, the interpolations made from their data and the continuous line that we define as shoreline. 

The visualization is controlled with the variables: 
View=1; (in case we enable viewing) 
Points=1; (in case we enable the representation of the points of each marker) 
Splines=1; (in case we enable the representation of the splines of each marker) 
Shoreline=1; (in case we enable shoreline representation) 

#3 The splitting of the images randomly into training and test groups.

#4 Preparing the images by decomposing them into columns and constructing the target vectors 

#5 Define the structure of the network 

#6 Define training parameters 

#7 Train the network (optionally a previously trained network can be upload) 

#8 Perform the test 

#9 Present the image-by-image results in both the test and training groups

#10 shows a table to summary of the results

![image](https://user-images.githubusercontent.com/62955998/214618649-24e289b2-8401-4746-957e-b6231e9f1821.png)
