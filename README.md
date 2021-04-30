# Good vs Evil: Comic Character Classification

In this project, we created a balanced image dataset of comic book characters labeled “superhero” or “villain” through automated search engine scraping. 
Our primary task is to use a CNN model to test the strength of our curated dataset, our secondary task is to compare the accuracy of different CNN approaches.

This project explores both the challenges of scraping google image search results for dataset creation as well as image-based classification of comic characters as 
“superhero” vs. “villain”. 

We experiment with classifying unseen comic book characters based on their images by: (1) building our own CNN using classic architecture and 
(2) using ResNet for transfer learning with pretrained models. 
To improve model performance we also experiment with: (1) refining the models’ structure and hyperparameters and (2) expanding our image search results to augment the dataset
