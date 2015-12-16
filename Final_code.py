
import os,glob
from nltk import word_tokenize, pos_tag, pos_tag_sents
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import RandomizedPCA
from sklearn.cross_validation import StratifiedShuffleSplit
import numpy as np
import sys
from scipy.sparse import hstack, vstack, csr_matrix

from networkx.algorithms.shortest_paths.unweighted import predecessor
sys.path.append('F:/box/Box Sync/kaggle/xgboost')
#import xgboost as xgb
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from collections import Counter
import time

#[3040]    train-error:0.078547    val-error:0.127958 with trigrams
#[2071]    train-error:0.052954    val-error:0.112273 bigrams and min df=3
#[2528]    train-error:0.030310    val-error:0.104018 bigrams min df3 tree deapth 6
#[3484]    train-error:0.000236    val-error:0.077050 trigrams min df2 tree, 1000 components
#[2934]    train-error:0.000354    val-error:0.073198 trigrams min df2 tree, 1000 components
class DocumentPrep(object):
    """
    The general class for handling unstructured, raw, textual documents
    assuming here, however, that raw files are one excerpt per line, for
    the sake of this project
    
    """
    def __init__(self, indir=None,testdir=None):
        self.indir=indir
        self.test_dir=testdir
        self.train_label=[]
        self.train_X=[]
        self.test_X=[]
        self.feature_dict=[]
    
    def extract_input(self):
        """
        use tf-idf weighted vector representation to represent the input data
        """
        raw_data=self.load_file_excerpts_raw(self.indir)
        self.test_X=self.load_file_excerpts_raw(self.test_dir)
        self.train_label=[int(raw[-1]) for raw in raw_data]
        if 1 in self.train_label:
            print 'yes'
        #:-3 is deleting the end of the sentence which is not always a period
        self.train_X=[raw[:-2] for raw in raw_data]

        vectorizer=TfidfVectorizer(ngram_range=(1,3),min_df=2,sublinear_tf=True,lower_case=False)#stop_words='english',min_df=3)
        #ct_vectorizer=CountVectorizer(ngram_range=(1,2),stop_words='english')
        sparse_X=vectorizer.fit_transform(self.train_X+self.test_X)
        #sparse_X_ct=ct_vectorizer.fit_transform(self.train_X[1])
        self.feature_dict=vectorizer.get_feature_names()
        #print len(ct_vectorizer.get_feature_names())
        print 'Feature dict length: ' + str(len(self.feature_dict))
        print len(vectorizer.get_feature_names())
        #print sparse_X_ct.shape
        return (sparse_X, len(self.train_X))

    def extract_input_test(self):
        """
        use tf-idf weighted vector representation to represent the input data
        """
        raw_data=self.load_file_excerpts_raw(self.indir)
        test_data=self.load_file_excerpts_raw(self.test_dir)

        vectorizer=TfidfVectorizer(ngram_range=(1,3),min_df=2,sublinear_tf=True,lower_case=False)#stop_words='english',min_df=3)
        sparse_X=vectorizer.fit_transform(raw_data + test_data)
        print len(vectorizer.get_feature_names())
        return (sparse_X, len(raw_data) )

    def get_pos(self, data, out_file):
        f_out = open(out_file, 'w')
        print len(data)
        count = 0
        for line in data:
            print count
            count += 1
            tokens = self.standardize(line)
            pos_sent = pos_tag(tokens)
            pos_line = [word + "/" + pos for word, pos in pos_sent]
            pos_line = ' '.join(pos_line)
            f_out.write(pos_line.encode('utf8') + "\n")

    def pos_tagger(self):
        #Tags each token in the input corpus with its corresponding part of speech
        #train_pos = self.get_pos(self.train_X, 'train_post.txt')
        #pickle.dump( train_pos, open( "train_pos.p", "wb" ) )
        test_data=self.load_file_excerpts_raw(self.test_dir)
        test_pos = self.get_pos(test_data, 'test_pos.txt')

        vectorizer=TfidfVectorizer(ngram_range=(1,3),min_df=2,sublinear_tf=False)#stop_words='english',min_df=3)
        sparse_X=vectorizer.fit_transform(train_pos + test_pos)
        pos_s=vectorizer.get_feature_names()
        print pos_s
        
        return
    
    def xgboost_pred(self,train,labels,test,test_labels,final_test):
        params = {}
        params['objective'] = 'binary:logistic'
        params["eval_metric"]="error"
        params["eta"] = 0.01 #0.02 
        params["min_child_weight"] = 6
        params["subsample"] = 0.9 #although somehow values between 0.25 to 0.75 is recommended by Hastie
        params["colsample_bytree"] = 0.7
        params["scale_pos_weight"] = 1
        params["silent"] = 1
        params["max_depth"] = 8
        #params["num_class"]=8
        
        plst = list(params.items())
        
        #Using 5000 rows for early stopping. 
        #offset = len(labels)/6
        #print offset
        num_rounds = 20000
        xgtest = xgb.DMatrix(final_test)
        
        xgtrain = xgb.DMatrix(train, label=labels)
        xgval = xgb.DMatrix(test, label=test_labels)
        
        watchlist = [(xgtrain, 'train'),(xgval, 'val')]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=300)
        #create a train and validation dmatrices 

        #reverse train and labels and use different 5k for early stopping. 
        # this adds very little to the score but it is an option if you are concerned about using all the data. 
#     train = train[::-1,:]
#     labels = np.log(labels[::-1])
# 
        print 'ready to generate test data'

        #combine predictions
        #since the metric only cares about relative rank we don't need to average
        return model.predict(xgtest,ntree_limit=model.best_iteration)
    
    
    def sentence_length_count(self):
        """
        corpus is a list of excerpts, one line each
        this method actually finds average number of sentences in each excerpt
        
        """
        for exc in self.train_X[:20]:
            print exc.split('.')
        
        #print [[sent for sent in exc.split('.')] for exc in self.train_X ]
        test = [np.mean(len([sent for sent in exc.split('.')])) for exc in self.train_X ]
        test.extend([np.mean(len([sent for sent in exc.split('.')])) for exc in self.test_X ])
        print 'average sentence length is '+str(test[5])
        print np.any(np.isnan(test))
        return test
    
    def avg_word_in_sent(self):
        test=[np.mean([len(sent) for sent in exc.split('.') ]) for exc in self.train_X]
        test.extend([np.mean([len(sent) for sent in exc.split('.')]) for exc in self.test_X])
        print 'average sentence length is '+str(test[5])
        
        print np.any(np.isnan(test))
        return test
    
    
    
    def avg_word_length(self):
        """
        """
        test=[np.mean([len(sent) for sent in exc.split(' ')]) for exc in self.train_X]
        print test[3]
        test.extend([np.mean([len(sent) for sent in exc.split(' ')]) for exc in self.test_X])
        print np.any(np.isnan(test))
        return test
    
    def word_richness(self):
        
        word_freq=[len(dict(Counter(exc))) for exc in self.train_X]
        word_freq.extend([len(dict(Counter(exc))) for exc in self.test_X])
        print 'word richness of the excerpt is '+str(word_freq[5])
        return word_freq
    
    def train_classifer_xgb(self,X):
        """
        train an xgb classifier 
        """
        return
        
    def get_all_files(self,directory):
        """ files within the given directory.
        Param: a string, giving the absolute path to a directory containing data files
        Return: a list of relative file paths for all files in the input directory.
        """
        filelist=glob.glob(directory+'/*.txt')
        return filelist
    
    def flatten(self,l):
        """
        input: n+1 dimensional list
        output: n dimensional list
        """
        return [subl for item in l for subl in item]
    
    def standardize(self,rawexcerpt):
        """
        input: rawexcerpt, one str of excerpt
        output: list of str, tokens in this excerpt
        """
        return word_tokenize(rawexcerpt.decode('utf8'))
    
    def load_file_excerpts_raw(self,filepath):
        """
        same as load_file_excerpts, except without tokenizing into a list of str
        via nltk.tokenizer
        """
        f=open(filepath)
        return[line.strip() for line in iter(f)]
    
    def load_file_excerpts(self,filepath):
        """that takes an absolute filepath and returns a list of
all excerpts in that file, tokenized and converted to lowercase. Remember that in our data files, each line
consists of one excerpt.
Param: a string, giving the absolute path to a file containing excerpts
Return: a list of lists of strings, where the outer list corresponds to excerpts and the inner lists
correspond to lowercase tokens in that exerpt
        """
        print 'loading '+filepath
        f=open(filepath)

        allexecerpts=[self.standardize(line)for line in iter(f)]
        print 'number of excerpts in this file is '+str(len(allexecerpts))
        return allexecerpts
        
    
    def load_directory_excerpts(self,dirpath):
        """takes an absolute dirpath and returns a list
of excerpts (tokenized and converted to lowercase) concatenated from every file in that directory.
Param: a string, giving the absolute path to a directory containing files of excerpts
Return: a list of lists of strings, where the outer list corresponds to excerpts and the inner lists
correspond to lowercase tokens in that exerpt
        """
        final_list=self.flatten([self.load_file_excerpts(files) for files in self.get_all_files(dirpath)])
        print 'number of excerpts in the entire path is '+str(len(final_list))
        return final_list

    def write_pred_file(self,out_path,pred):
        """
        input: str, the directory where the 
        pred: numpy array
        """
        with open(out_path,'w') as f:
            ct=0
            for out in pred:
                print out
                if out==1:
                    ct=ct+1
                f.write(str(out))
                f.write('\n')

    def get_pos_count(self, pos_tags_file, pos_dict):
        #get average count of each pos per excerpt
        pos_excerpts = self.load_file_excerpts_raw(pos_tags_file)
        all_pos = set([])
        for line in pos_excerpts:
            sents = line.split('.')
            cur_count = Counter([])
            for sent in sents:
                pos_tok = sent.split()
                all_pos.update(pos_tok)
                cur_count += Counter(pos_tok)
            count_dict = dict(cur_count)
            for key in pos_dict.keys():
                if key in count_dict.keys():
                    pos_dict[key].append(count_dict[key]/float(len(sents)))
                else:
                    pos_dict[key].append(0)
        return pos_dict

    def wordcount_helper(self, data, word_list, row, col, vals):

        cur_row = 0
        if row:
            cur_row = max(row) + 1

        for line in data:
            if cur_row % 1000 == 0:
                print cur_row
            words = word_tokenize(line.decode('utf-8'))
            word_dict = dict(Counter(words))
            for key in word_dict.keys():
                if key not in word_list:
                    word_list.append(key)
                row.append(cur_row)
                col.append(word_list.index(key))
                vals.append(word_dict[key])
            cur_row += 1
        return word_list, row, col, vals

    def get_wordcount(self, train_doc, test_doc):
        train_data = self.load_file_excerpts_raw(train_doc)
        word_list, row, col, vals = self.wordcount_helper(train_data, [], [], [], [])
        test_data = self.load_file_excerpts_raw(test_doc)
        word_list, row, col, vals = self.wordcount_helper(test_data, 
            word_list, row, col, vals)
        
        f_vals = open('vals.txt','w')
        np.save(f_vals, vals)
        f_row = open('row.txt','w')
        np.save(f_row, row)
        f_col = open('col.txt','w')
        np.save(f_col, col)
        return create_csr(vals, row, col)

    def get_wc_saved(self):
        vals = np.load('vals.txt')
        row = np.load('row.txt')
        col = np.load('col.txt')
        return create_csr(vals, row, col)

def create_csr(new_col, row_a, col_a, ):
    row = np.array(row_a)
    col = np.array(col_a)
    data = np.array(new_col)
    new_f = csr_matrix((data, (row, col)), shape=( max(row_a) + 1, max(col_a) + 1))
    return new_f


def add_feature(sparse_x, new_col):
    #Adds new feature column to current feature matrix
    new_f = create_csr(new_col, range(len(new_col)), [0]*len(new_col))
    return hstack([sparse_x, new_f]).tocsr()

def add_pos_feature(doc_prep, data_test):
    pos_dict = {'PRP$':[], 'WDT':[], 'JJ':[], 'WP':[],'RP':[],'$':[],'(':[],'FW':[],',':[],
                'PRP':[],'RB':[],'NNS':[],'NNP':[], 'WRB':[],'RBS':[],'EX':[],'MD':[],'UH':[],
                'VBG':[],'VBD':[],'VBN':[],'VBP':[],'VBZ':[], 'NN':[],'CC':[],'PDT':[],
                'CD':[],'WP$':[],'JJS':[],'JJR':[],"''":[], 'DT':[],')':[],'POS':[],'TO':[],
                'VB':[],'RBR':[],'IN':[],'NNPS':[]}
    pos_dict = doc_prep.get_pos_count(local_dir + "/CIS530/pos_tagged/only_pos_train",pos_dict)
    pos_dict = doc_prep.get_pos_count(local_dir + "/CIS530/pos_tagged/only_pos_test",pos_dict)
    for key in pos_dict.keys():
        data_test = add_feature(data_test, pos_dict[key])
    return data_test





if __name__=="__main__":

    local_dir = "C:/Users/ze/Box Sync/CIS 530/final_project"

    #Original text:
    doc_prep=DocumentPrep(indir=local_dir + '/train/project_articles_train',testdir= local_dir + '/test/project_articles_test')
    

    #doc_prep.get_wordcount(local_dir + '/CIS530/pos_tagged/train_pos')

    (data,train_len)=doc_prep.extract_input()
    print type(data)
    labels=np.asarray(doc_prep.train_label)
    sent_len=np.asarray(doc_prep.sentence_length_count())
    avg_word_len=np.asarray(doc_prep.avg_word_length())
    avg_sent_len=np.asarray(doc_prep.avg_word_in_sent())
    print avg_word_len.shape
    print len(sent_len)


    train=data[:train_len,:]
    test=data[train_len:,:]
    #POS Tagged text:
    doc_prep_pos=DocumentPrep(indir=local_dir + '/train/project_articles_train',testdir= local_dir + '/test/project_articles_test')
    #doc_prep_pos=DocumentPrep(indir=local_dir + '/data/project_articles_train',testdir= local_dir + '/data/project_articles_test')
    
    (data_pos,train_len_test)=doc_prep_pos.extract_input_test()
    print data_pos.shape
    #Add sentence length feature
    added_features=np.transpose(np.vstack([sent_len,avg_sent_len,avg_word_len]))
    stder=StandardScaler(with_mean=True)
    added_features=csr_matrix(stder.fit_transform(added_features))
    print added_features.shape
    data_test=hstack([data_pos,added_features]).tocsr()
    #data_test = add_feature(data_pos, avg_word_len)
    print data_test.shape
    
    


    
    train_test=data_test[:train_len_test,:]
    test_test=data_test[train_len:,:]

    
    
    
    c_val=40
    sss = StratifiedShuffleSplit(labels, 3, test_size=0.3)
    diff = []
    for train_index, test_index in sss:
        X_train, X_test = train[train_index], train[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        clf=svm.LinearSVC(C=c_val)
        clf.fit(X_train,y_train)
        print c_val
        print 'Baseline Score: ' + str(clf.score(X_test,y_test))
 
        X_train_t, X_test_t = train_test[train_index], train_test[test_index]
        clf_test=svm.LinearSVC(C=c_val)
        clf_test.fit(X_train_t,y_train)
        print 'Test Score: ' + str(clf_test.score(X_test_t,y_test))
 
        cur = clf_test.score(X_test_t,y_test) - clf.score(X_test,y_test)
        diff.append(cur)
        print 'Difference: ' + str(cur)
        c_val=c_val+30


    clf=svm.LinearSVC(C=70)
    clf.fit(train_test,labels)
    pred=clf.predict(test_test)
    doc_prep.write_pred_file('test_new_features.txt', pred)
