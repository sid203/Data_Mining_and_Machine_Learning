#Load Dataset 
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer
from nltk import word_tokenize
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import pprint as pp
import string
from sklearn.neighbors import KNeighborsClassifier



cachedStopWords = stopwords.words("english")
def testFuncNew(text):
        text = ' '.join([word for word in text.split() if word not in cachedStopWords])
        return text

############################################
stemmer = EnglishStemmer()

def stemming_tokenizer(text):
    #text =text.decode('utf-8')
    stop_words_removed_text = testFuncNew(text)
    stemmed_text = [stemmer.stem(word) for word in word_tokenize(stop_words_removed_text, language='english')]
    return stemmed_text


def stemming_tokenizer_no_stop_words(text):
    #text =text.decode('utf-8')
    stemmed_text = [stemmer.stem(word) for word in word_tokenize(text, language='english')]
    return stemmed_text
######################################################################




## Dataset containing Positive and neative sentences on Amazon products
data_folder_training_set = "./Ham_Spam_comments/Training"
data_folder_test_set     = "./Ham_Spam_comments/Test"

training_dataset = load_files(data_folder_training_set)
test_dataset = load_files(data_folder_test_set)
print()
print ("----------------------")
print((training_dataset.target_names))
print ("----------------------")
print()


# Load Training-Set
X_train, X_test_DUMMY_to_ignore, Y_train, Y_test_DUMMY_to_ignore = train_test_split(training_dataset.data,
                                                    training_dataset.target,
                                                    test_size=0.0)
target_names = training_dataset.target_names

# Load Test-Set
X_train_DUMMY_to_ignore, X_test, Y_train_DUMMY_to_ignore, Y_test = train_test_split(test_dataset.data,
                                                    test_dataset.target,
                                                    train_size=0.0)

target_names = training_dataset.target_names
print()
print("----------------------")
print ("Creating Training Set and Test Set")
print()
print ("Training Set Size")
print(Y_train.shape)
print()
print ("Test Set Size")
print(Y_test.shape)
print()
print("Classes:")
print(target_names)
print ("----------------------")

## Vectorization object
vectorizer = TfidfVectorizer(strip_accents= None,preprocessor = None)

## classifier
nbc = KNeighborsClassifier()

pipeline = Pipeline([('vect',vectorizer),('nbc',nbc),])

parameters = {
    'vect__tokenizer': [stemming_tokenizer,stemming_tokenizer_no_stop_words],
    'vect__ngram_range': [(1, 1),(1,3)],
    'vect__use_idf':[True,False],
    'nbc__weights':['uniform','distance'],
    'nbc__algorithm':['ball_tree','kd_tree']
    
    }

grid_search = GridSearchCV(pipeline,
                           parameters,
                           scoring=metrics.make_scorer(metrics.average_precision_score, average='weighted'),
                           cv=10,
                           n_jobs=-1,
                           verbose=10)

grid_fit=grid_search.fit(X_train, Y_train)
grid_fit.grid_scores_

print()
print("Best Estimator:")
pp.pprint(grid_search.best_estimator_)
print()
print("Best Parameters:")
pp.pprint(grid_search.best_params_)
print()
print("Used Scorer Function:")
pp.pprint(grid_search.scorer_)
print()
print("Number of Folds:")
pp.pprint(grid_search.n_splits_)
print()



#Evaluation
Y_predicted = grid_search.predict(X_test)

# Evaluate the performance of the classifier on the original Test-Set
output_classification_report = metrics.classification_report(
                                    Y_test,
                                    Y_predicted,
                                    target_names=target_names)
print()
print ("----------------------------------------------------")
print(output_classification_report)
print ("----------------------------------------------------")
print()

# Compute the confusion matrix
confusion_matrix = metrics.confusion_matrix(Y_test, Y_predicted)
print()
print("Confusion Matrix: True-Classes X Predicted-Classes")
print(confusion_matrix)
print()

#Mathews_core coeff,#Normalized accuracy 

print('matthews_corrcoef:',metrics.matthews_corrcoef(Y_test,Y_predicted))
print('Normalized accuracy:',metrics.accuracy_score(Y_test,Y_predicted))




