# LLM Project

## Project Task
I chose the Topic Modelling task, which involves taking text from the 20_newsgroups dataset and sorting it into groups based on the text's content. 

## Dataset
The 20_newsgroups dataset comes presplit into training and testing sets, both prelabeled with true categories, both as a numerical value and a string. These string values reveal that the categories are the end result of breaking down larger, less specific categories into more granular ones. There are 11.3k items in the training set, and 7.5k in the test set, with a nearly uniform distribution across the 20 categories present.

## Pre-trained Model
The pretrained model I used was DistilBert, and more specifically the DistilBertForSequenceClassification. It has the paired tokenizer DistilBertTokenizer, which I used for tokenization prior to training the model. Both were initialized using the pretrained instance 'distilbert-base-uncased'. 

I chose this model as, from the ones we discussed, this was the one suited to the task I had chosen, which was confirmed when it showed up on the HuggingFace directory when I searched by task. I then chose the offshoot version of the model because it is very specifically suited to my task.

## Performance Metrics
Because this is a classification task, to the best of my knowledge and as far as I can find, the model by default uses a cross-entropy loss function to evaluate its progress. This only happens if it is given labeled data during training, but my dataset *is* labeled, so it was produced. 

Given I have 7 labels, random guessing would give a loss score of around 1.95, while the model had a training loss of only 0.81 and a final training loss of 0.52 on the train set and 0.63 on the test set. 

I also used accuracy and f1, as with classification the goal is to be correct. On evaluation, the accuracy scores were 82% to 77%, and the f1 was 0.80 to 0.76. Because I altered the number of categories I was working with and created under-represented categories, it was important to also have the f1 score, as it is potentially more sensitive to imbalanced data. However, both metrics show similar performance and similar drops between the training and testing sets. This model, even with very basic parameters and one round of training, performs quite well; miles better than the SKLearn attempt, and also than complete randomness, which would be 1/7 or about 14%.

Getting the accuracy and f1 also produced a new pair of loss values, and I am not entirely sure if those are for the trained model, while the others were for the in-training model, or if something else was going on. But the loss scores produced by the evaluate trainer method showed as lower. This makes sense if they are from the trained model. 

## Hyperparameters
I did very little parameter tuning on the actual model that I ran, both as a result of the initial version that ran in collab taking 7 hours to run, and because I still don't have a very good understanding of what the various parameters do. Most of the parameters on the tokenzier didn't seem to apply in my case, or where already the way I needed them by default. The only task-specific parameter I passed to the model was the number of labels, and that wouldn't change with tuning. I don't think I know enough to say what parameters I could tune to get better results, and that is likely because I don't know that they exist.  

## Relevant Links
[Model](https://huggingface.co/distilbert/distilbert-base-uncased)
[Dataset](https://huggingface.co/datasets/SetFit/20_newsgroups)
