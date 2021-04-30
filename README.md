# Good vs Evil: Comic Character Classification

This project explores both the challenges of scraping google image search results for dataset creation as well as image-based classification of comic characters as 
“superhero” vs. “villain”. In this project, we created a balanced image dataset of comic book characters labeled “superhero” or “villain” through automated search engine scraping. Our primary task is to use a CNN model to test the strength of our curated dataset, our secondary task is to compare the accuracy of different CNN approaches.

We experiment with classifying unseen comic book characters based on their images by: (1) building our own CNN using classic architecture and 
(2) using ResNet for transfer learning with pretrained models. 
To improve model performance we also experiment with: (1) refining the models’ structure and hyperparameters and (2) expanding our image search results to augment the dataset

![Display training images](https://user-images.githubusercontent.com/6591820/116647976-f569f200-a949-11eb-8789-c1cfea2f804d.png)

## Overview of Program Structure

* *download.py* - Performs a Google image search and downloads the resulting images to the output directory.

* *desirability_filter.py* - Program that filters out noisy Google image search results, so that our 
“superhero” vs. “villain” classifier will only be trained on desirable images scraped from the web.

* *DesirabilityResNetClassifier.pth* - Model weights for our preliminary “desirable” vs. “undesirable” image classifier that filters out noisy images. 

* *classifier.py* - Program that loads in desirable image data, and trains a CNN to classify images as "superhero" or villain". Transfer learning with 
ResNet18 is leveraged and the model with the highest validation accuracy is chosen so the model can generalize well.

*  *pipeline.py* - Program that automates the process of loading a dataset, downloading the images, filtering undesirable images, training the
CNN classifier, and producing the resulting scoring metrics. 

* *cfgspec.yaml* - The configuration specification 'cfgspec' is a dictionary that will be used to generate a list containing valid permutations of input parameters.
All possible permutations of the values given for the parameters will be added to the test list, unless explicitly suppressed.

* *run_batch.py* - Helper file that reads in the YAML input and outputs a CSV file containing the scoring metrics. 

### Getting Started
To start our program, use the command: *python3 runbatch.py -n 100 -o metrics.csv cfgspec.yaml*
*Note:* This will read in the YAML file 'cfgspec.yaml', download 100 images per character, run through the full Pipeline process, and output a CSV file titled 'metrics.csv'
