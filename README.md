# money-bot-chatbot-classifier

## Model: 
This model is kind of chatbot plus classifier.
The main architecure of this model is WORD CHARACTER -> ACTIVITY classifier.
Why character Not word?
My thought is if user mispell any word so that our chatbot can detect that error.

## To run this file:
$ python -m trainer.train    --base_dir=${PWD}    --data_dir='data'    --local_file_name='data.csv'     --external_file_name='external_chat.csv'    --resources_dir='resources'    --output_dir='output'

You can set optional training hyperparameters. (List of command line parameters are defind in trainer.py file)
