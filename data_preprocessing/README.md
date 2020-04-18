# prior data feature analysis
python my_analyze_data_00.py --vocab 1.txt --npy 1.npy

# For data preprocessing
## 1. word tokenization with stanford corenlp
python3 my_eng_word_tokenize_1.py
## 2. clean the word tokenization results
python3 my_tokenize_to_input_2.py
## 3. separate cleaned data as "premise" and "hypothesis"(as in textual entailment), save to files. It concatenates word-level and tweet-level features behind the original sentence.
python my_get_tweet_text_3.py --vocab 1.txt --npy 1.npy

# For ESIM
## 4. create word embeddings(rumor-vocab-19.txt) in ESIM
cd /codes/rumor-ESIM-88-branch/
nohup python3 snli_train_lr.py --save_path all_model_saved/model_saved_0121_a_1 >log/log0121_a_1
## 5. transfer created word embedding to 300-dimensional Google News embedding (rumor-news-embedding-19.npy)
cd /data/glove/
python my_create_embedding_4.py --vocab=rumor-vocab-19.txt --npy=rumor-news-embedding-19.npy

# For data augmentation
# expand data on unbalanced labels - support, deny & query
my_enlarge_dataset_5.py, my_preprocess_enlarge_dataset_6.py
