import os,glob
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import RandomizedPCA
from sklearn.cross_validation import StratifiedShuffleSplit
import numpy as np
import sys
from networkx.algorithms.shortest_paths.unweighted import predecessor
sys.path.append('F:/box/Box Sync/kaggle/xgboost')
import xgboost as xgb
from sklearn import svm


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
        self.feature_dict=[]
    
    def extract_input(self):
        """
        use tf-idf weighted vector representation to represent the input data
        """
        raw_data=self.load_file_excerpts_raw(self.indir)
        test_data=self.load_file_excerpts_raw(self.test_dir)

        self.train_label=[int(raw[-1]) for raw in raw_data]
        if 1 in self.train_label:
            print 'yes'
        self.train_X=[raw[:-3] for raw in raw_data]
        print self.train_X[4]
        print test_data[2]
        print len(self.train_X)
        print len(test_data)

        vectorizer=TfidfVectorizer(ngram_range=(1,3),min_df=2,sublinear_tf=False)#stop_words='english',min_df=3)
        #ct_vectorizer=CountVectorizer(ngram_range=(1,2),stop_words='english')
        sparse_X=vectorizer.fit_transform(self.train_X+test_data)
        #sparse_X_ct=ct_vectorizer.fit_transform(self.train_X[1])
        self.feature_dict=vectorizer.get_feature_names()
        #print len(ct_vectorizer.get_feature_names())
        print len(self.feature_dict)
        #print sparse_X_ct.shape
        return (sparse_X, len(self.train_X))
    
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
    
    
    def sentence_length_count(self,corpus):
        """
        corpus is a list of excerpts, one line each
        
        """
        return [avg([sent for sent in exc.split('.')]) for exc in corpus ]
        
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
        with open(out_path) as f:
            for out in pred:
                f.write(str(out))
                f.write('\n')

if __name__=="__main__":
    doc_prep=DocumentPrep(indir='F:/box/Box Sync/CIS 530/final_project/train/project_articles_train',testdir='F:/box/Box Sync/CIS 530/final_project/test/project_articles_test')
    (data,train_len)=doc_prep.extract_input()
    labels=np.asarray(doc_prep.train_label)
    print type(data)
#     pca=RandomizedPCA(n_components=1000)
#     pca.fit(data)
#     transformed_data=pca.transform(data)
#     train=transformed_data[:train_len,:]
    train=data[:train_len,:]
    
    print train.shape
    print train_len
    test=data[train_len:,:]
    
    c_val=37
    sss = StratifiedShuffleSplit(labels, 9, test_size=0.3)
    for train_index, test_index in sss:
     X_train, X_test = train[train_index], train[test_index]
     y_train, y_test = labels[train_index], labels[test_index]
     clf=svm.LinearSVC(C=39)
     clf.fit(X_train,y_train)
     print c_val
     print clf.score(X_test,y_test)
     
     c_val=c_val+2
    
