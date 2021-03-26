README: 

The collections of files provided in this GitHub enviornment were written for
homework 2 in Dr. Feng Luo's Deep Learning Course at Clemson University during
the Spring 2021 semester (CPSC 8430). 

The files in this github are the following: 
seqtoseq.py - All code that was implemented for sequence to sequence model
HW2.pdf - Assignment prompt written by Dr. Feng Luo
output.txt - output file containing video id and output caption
readme.md - The current open file 
report.pdf - Report on project 

To run these files, the following libraries are required:
python 3.8
Pytorch 1.7
Numpy 1.19
json 2.0.9

The dataset required to run this project is the dataset provided by
Dr. Luo in the original prompt. To my knowledge, students were requested
to avoid uploading this dataset to github for space reasons. The 
seqtoseq.py file contains my implementation of the sequence-to-sequence
model needed for this project. The code will compile and run, but it does
not properly implement the encoding-decoding process required by the 
direction prompt. To compile this code, you will need to change the path
variables in lines 16, 17, 30 to point to the location in which the dataset
is saved locally. The program produces an output.txt file that contains 
the testing videos and their captions. However, these captions are not
produced by my model, but rather read directly from the json file from which
they came and piped into this output file. 

There were two main portions of the project that caused me problems. 
The first was the presence of an encoder. The class was told approximately 
1-2 weeks into the project that we did not need to implement an encoder, 
only the decoder. As a result, I spent two weeks trying to understand
how to pass the video features directly to the decoder, believeing that
the video features represented the context vector output from an encoder. 
It was only in the week of the project being due that the TA set me 
straight in that encoding was required. The second problem pretained to 
the embedding of captions. The process of encoding captions into vectors, 
embedding them into a matrix, properly padding the matrix, and passing this 
to the model was confusing. I struggled to find a resource that explained
how to implement this process. In general, every portion of this project 
was novel to me, and apart from bombarding the TA with questions via email, 
I did not know where to turn for help in implementation, considering that
this type of help is not provided during office hours (which focuses on
big picture or theory based questions). 

The portions of the project I could complete were the creation of a 
vocabulary using the provided captions, creating the necessary data 
structures from the provided training and testing data, and writing a 
seqtoseq model that I believe has the required architecture. My hope 
was to receive partial credit for the portions of the code I could finish, 
but I am aware that since the code does not do what was required by 
the rubric, this will not be much if any.

The seqtoseq class implemented in my seqtoseq.py file used the following 
resource as a blueprint: 
https://github.com/YiyongHuang/S2VT/blob/master/module.py


Note to TA: Thank you for the time you spent responding to my constant
email questions and attempting to provide resources that would help me. 
I apologize that I could not produce a more complete project given all the
effort you put into attempting to assist me.  
