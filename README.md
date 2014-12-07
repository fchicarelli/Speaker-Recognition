#Speaker-Recognition using Gaussian Mixture Models

##Overview

Final project for the course ***ECE446 - Sensory Communication*** at ***University of Toronto***. The project consists in a Speaker Recognition system that uses Gaussian Mixture Models (GMM) and a report that explains the entire system can be found [here](https://github.com/fchicarelli/Speaker-Recognition/blob/master/Final%20Report.pdf).

## Dependencies

The system has dependencies with the following libraries:
* Scipy
* Numpy
* Scikit-Learn

## Database

First it is needed to build a database of users. Voice samples of each user in the database are recorded and saved as .wav files at `./Database/<username>/`, where \<username\> is the name of each user. In this case, more samples means more accuracy.

The samples are text-independent, i.e. the user can say anything and the system will still work.

## How to run

The file that need to be run is the `extract.py`. It will uses the file at `./Test/` as the one to be recognized. As the program start running, appropriate outputs appear showing the results.

## Observations 

The code is not well organized, and it needs to be improved. This probably will be solved in the future, when I have free time. 
