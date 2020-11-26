import keras
from keras.utils import to_categorical
import numpy as np
import os
import pickle as pkl

needReport = False   

train_dict = pkl.load(open("data/train.pkl", "rb"))
val_dict = pkl.load(open("data/val.pkl", "rb"))
test_dict = pkl.load(open("data/test.pkl", "rb"))

vocab_dict = {'_unk_': 0, '_w_pad_': 1}

for doc in train_dict['word_seq']:
    for word in doc:
        if(word not in vocab_dict):
            vocab_dict[word] = len(vocab_dict)

tag_dict = {'_t_pad_': 0} # add a padding token

for tag_seq in train_dict['tag_seq']:
    for tag in tag_seq:
        if(tag not in tag_dict):
            tag_dict[tag] = len(tag_dict)
word2idx = vocab_dict
idx2word = {v:k for k,v in word2idx.items()}
tag2idx = tag_dict
idx2tag = {v:k for k,v in tag2idx.items()}            

print("size of word vocab:", len(vocab_dict), "size of tag_dict:", len(tag_dict))

# Provided function to test accuracy
# You could check the validation accuracy to select the best of your models
def calc_accuracy(preds, tags, padding_id="_t_pad_"):
    """
        Input:
            preds (np.narray): (num_data, length_sentence)
            tags  (np.narray): (num_data, length_sentence)
        Output:
            Proportion of correct prediction. The padding tokens are filtered out.
    """
    preds_flatten = preds.flatten()
    tags_flatten = tags.flatten()
    non_padding_idx = np.where(tags_flatten!=padding_id)[0]
    
    return sum(preds_flatten[non_padding_idx]==tags_flatten[non_padding_idx])/len(non_padding_idx)

train_x = train_dict['word_seq'] 
train_y = train_dict ['tag_seq'] 
valid_x = val_dict ['word_seq']
valid_y = val_dict[ 'tag_seq'] 
test_x = test_dict['word_seq']
print(np.array(train_x).shape , np.array(train_y).shape, np.array(valid_x).shape, np.array(valid_y).shape)




import kashgari
from kashgari.tasks.labeling import BiGRU_Model, BiGRU_CRF_Model, BiLSTM_CRF_Model,  BiLSTM_Model ,CNN_LSTM_Model 
# from kashgari.embeddings import TransformerEmbedding
import os 

def getModel(name ) : 
    return {
        'BiGRU_Model' : BiGRU_Model () , 
        'BiGRU_CRF_Model' : BiGRU_CRF_Model() , 
        'BiLSTM_CRF_Model' : BiLSTM_CRF_Model ( ) , 
        'BiLSTM_Model' : BiLSTM_Model() , 
        'CNN_LSTM_Model' : CNN_LSTM_Model() , 
    }[name]

def getModelClass(name): 
    return {
        'BiGRU_Model' : BiGRU_Model , 
        'BiGRU_CRF_Model' : BiGRU_CRF_Model, 
        'BiLSTM_CRF_Model' : BiLSTM_CRF_Model, 
        'BiLSTM_Model' : BiLSTM_Model , 
        'CNN_LSTM_Model' : CNN_LSTM_Model , 
    }[name]

models = ['BiGRU_Model' , 'BiGRU_CRF_Model', 'BiLSTM_CRF_Model' , 'BiLSTM_Model' ,'CNN_LSTM_Model' ]

import pandas as pd 
import numpy as np 
import json 
from evaluate import evaluate

# df = pd.DataFrame(columns = ["model" , "embedding" ] )
# df2 = none
# d = {'col1': [1], 'col2': [3]}
# df = pd.DataFrame(data =d )
df = pd.DataFrame(columns = ['model' , 'epoch' , 'batch_size' , 'max_f1_score' , 'which_maxf1' , 'support_maxf1' , 'min_f1_score' , 'which_minf1' , 'support_minf1' , 'macro_avg_f1', 'train_acc' , 'val_acc'] )
print(df)

for e in [5 , 10  ] : 
    for b in [64, 128 ] : 
        for m in models : 
            # for em in embeddings : 
                model_folder = f'{m}_{b}_{e}'
                if os.path.isdir(model_folder) : 
                    model = getModelClass(m) .load_model(model_folder)
                else : 
                    model = getModel(m)
                    model.fit(train_x, train_y , valid_x, valid_y , batch_size=b, epochs=e)
                    model.save(model_folder)


                train_preds = model.predict(train_x)
                val_preds = model.predict(valid_x)
                test_preds = model.predict(test_x)


                


                df2 = pd.DataFrame({'id': train_dict["id"],
                   'labels': [json.dumps(np.array(preds).tolist()) for preds in train_preds]})
                df2.to_csv(f'{model_folder}_train_preds.csv', index=False)
                df2 = pd.DataFrame({'id': val_dict["id"],
                'labels': [json.dumps(np.array(preds).tolist()) for preds in val_preds]})
                df2.to_csv(f'{model_folder}_val_preds.csv', index=False)
                df2 = pd.DataFrame({'id': test_dict["id"],
                'labels': [json.dumps(np.array(preds).tolist()) for preds in test_preds]})
                df2.to_csv(f'{model_folder}_test_preds.csv', index=False)

                evaluate_v

                if needReport : 
                    train_report = model.evaluate(train_x , train_y )
                    val_report = model.evaluate(valid_x , valid_y )
                    train_acc = calc_accuracy(np.array(train_preds) , np.array(train_y) ) 
                    val_acc = calc_accuracy(np.array(val_preds) , np.array(valid_y ) )

                    print() 
                    print(f"{model_folder} train preds : " , train_acc) 
                    print(f"{model_folder} valid pred : " , val_acc) 
                    print() 
                    # df = pd.DataFrame(columns = ['model' , 'epoch' , 'batch_size' , 'max_f1_score' , 'which_maxf1' , 'support_maxf1' , 'min_f1_score' , 'which_minf1' , 'support_minf1' , 'train_acc' , 'val_acc'] )

                    max_f1_score = 0  
                    which_max_f1 = "" 
                    support_max_f1 = 0  
                    min_f1_score = 1
                    which_min_f1 = '' 
                    support_max_f1 = 0 
                    for item in train_report['detail'] :
                        temp = train_report['detail'][item] 
                        if temp['f1-score'] > max_f1_score and item != '_t_pad_' : 
                            max_f1_score = temp['f1-score'] 
                            which_max_f1 = item
                            support_max_f1 = temp['support']
                        if temp['f1-score'] < min_f1_score and item != '_t_pad_' : 
                            min_f1_score = temp['f1-score']
                            which_min_f1 = item 
                            support_max_f1 = temp['support'] 


                    d = {
                        'model' : [m] , 
                        'epoch' : [e]  , 
                        "batch_size" : [b] , 
                        'max_f1_score' : [max_f1_score] , 
                        'which_maxf1' : [which_max_f1] , 
                        'support_maxf1' : [support_max_f1] , 
                        'min_f1_score' : [min_f1_score] , 
                        'which_minf1' : [which_min_f1] , 
                        'support_minf1' : [support_max_f1] ,
                        'macro_avg_f1' : [train_report['f1-score']],  
                        'train_acc' : [train_acc] , 
                        'val_acc' : [val_acc] 
                    }
                    tempdf = pd.DataFrame(data =d )
                    df = df.append(tempdf , ignore_index= True )
                    print(df )
                    print( '============================================================================================================')

# test_y = model.predict(test_x) 
# report = model.evaluate(test_x , real_test_y ) 

if needReport : 
    print(df)
    df.to_csv( "comparison_table.csv" ,index = False) 


