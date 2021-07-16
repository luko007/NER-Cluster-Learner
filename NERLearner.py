import os

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

import NERFileLoader
from ClusterMaker import ClusterMaker


class NERLearner:
    CHUNK_SIZE = 50000
    TEST_PER = 0.3

    NER_FILE = os.path.join('Data', 'ner_dataset.csv')
    ALL_SENTENCES = os.path.join('Data', 'all_sentences.csv')

    def __init__(self):
        self.twitterBrownCluster = ClusterMaker()
        self.twitterBrownCluster.load_twitter_clusters(self.twitterBrownCluster.TWITTER_FILE_NAME)
        # Init NER DataFrames
        self.df_1m = pd.read_csv(self.NER_FILE, encoding="ISO-8859-1")
        self.df_1m_brown = self.twitterBrownCluster.add_brown_clusters_as_feature(self.df_1m)
        self.df_broad_twitter_corpus = NERFileLoader.load_broad_twitter_corpus()
        self.df_broad_twitter_corpus_brown = self.twitterBrownCluster.add_brown_clusters_as_feature(
            self.df_broad_twitter_corpus, True)

    def preprocess_and_fit(self, df, learner, pop_O_from_classes = True):
        '''
        Given the dataset and learner object learn and test, the output is classification_report
        :param pop_O_from_classes:
        '''
        X, y = self.encode_and_return_X_y(df)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.TEST_PER, random_state=13)

        learner.fit(X_train, y_train)

        y_predict = learner.predict(X_test)
        classes = np.unique(y_predict).tolist()
        new_classes = np.unique(classes.copy()).tolist()
        if pop_O_from_classes:
            new_classes.pop()

        return classification_report(y_pred=y_predict, y_true=y_test, labels=new_classes, output_dict=False)

    def encode_and_return_X_y(self, df):
        '''
        Preprocess the data and ecode using OneHotEncoder, output the features matrix X and the tags vector y
        '''
        o = OneHotEncoder()
        if 'Sentence #' in df.columns.values:
            df['Sentence #'] = df['Sentence #'].fillna(method='ffill')
        df['Capitalize'] = np.where(df.Word.str.capitalize() == df.Word, 1, 0)
        # There should be no rows with NaN by now
        df.dropna(inplace=True)
        X = df.drop(NERFileLoader.TAG_COLUMN, axis=1)
        X = o.fit_transform(X)
        y = df.Tag.values
        return X, y

    # def learn_broad_twitter_corpus(self, df_train, df_test, learner):
    #
    #     o = OneHotEncoder()
    #
    #     # X_train = df_train.drop(NERFileLoader.TAG_COLUMN, axis=1)
    #     # X_test = df_test.drop(NERFileLoader.TAG_COLUMN, axis=1)
    #     # y_train = df_train[NERFileLoader.TAG_COLUMN].values
    #     # y_test = df_test[NERFileLoader.TAG_COLUMN].values
    #
    #     # o.fit_transform(pd.concat((df_train, df_test)))
    #     # X_train = o.transform(X_train)
    #     # X_test = o.transform(X_test)
    #
    #     # There should be no rows with NaN by now
    #     df = pd.concat((df_train, df_test))
    #     df.dropna(inplace=True)
    #
    #     X = df.drop(NERFileLoader.TAG_COLUMN, axis=1)
    #     X = o.fit_transform(X)
    #     y = df.Tag.values
    #     classes = np.unique(y).tolist()
    #     classes.pop()
    #
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.TEST_PER, random_state=13)
    #
    #     learner.fit(X_train, y_train)
    #
    #
    #     # X_test = o.fit_transform(X_test)
    #
    #     return classification_report(y_pred=learner.predict(X_test),
    #                                  y_true=y_test,
    #                                  labels=classes,
    #                                  output_dict=True)
    #
    # def preprocess(self):
    #     curr_df = self.df_1m[: 30000]
    #     self.df_1m[:30001] = curr_df = curr_df.fillna(method='ffill')
    #     X = curr_df.drop('Tag', axis=1)
    #
    #     h = HashingVectorizer(n_features=2 ** 18, alternate_sign=False)
    #     v = DictVectorizer(sparse=False)
    #
    #     self.X_test = v.fit_transform(
    #         X.to_dict('records'))  # this line throw memory ex when using all the data
    #     self.y_test = curr_df.Tag.values
    #
    #     for i in range(math.ceil(self.df_1m.shape[0] / self.CHUNK_SIZE)):
    #         curr_df = self.df_1m[self.CHUNK_SIZE * i: self.CHUNK_SIZE * (i + 1)]
    #         curr_df = curr_df.fillna(method='ffill')
    #         X = curr_df.drop('Tag', axis=1)
    #         # v = DictVectorizer(sparse=False)
    #
    #         X = v.fit_transform(X.to_dict('records'))  # this line throw memory ex when using all the data
    #         y = curr_df.Tag.values
    #         classes = np.unique(y).tolist()
    #         print("i is:" + str(i) + ", size of training: " + str(len(X)))
    #         # if i == 0:
    #         #     X, self.X_test, y, self.y_test = train_test_split(X, y, test_size=self.TEST_PER, random_state=13)
    #         yield X, y, classes
    #
    # def learner(self, learner):
    #     accuracy = []
    #     num_of_items = 0
    #     for X_train, X_test, y_train, y_test, classes in self.preprocess():
    #         num_of_items += len(X_train)
    #         learner.partial_fit(X_train, y_train, classes)
    #         accuracy.append((learner.score(self.X_test, self.y_test), num_of_items))
    #
    #     # remove the class O (it is the most common and not interesting)
    #     # new_classes = classes.copy()
    #     # new_classes.pop ()
    #     # print (classification_report (y_pred=learner.predict (X_test), y_true=y_test, labels=new_classes))
    #     return accuracy

    def run_learner(self):
        '''
        The main function of our experiment, run the different learners that we specified on different datasets
        '''
        learners = {'LinearSVC': LinearSVC(dual=True, loss='hinge', max_iter=4000)
                    # 'SGDClassifier': SGDClassifier(loss='modified_huber'),
                    # 'Passive-Aggressive': PassiveAggressiveClassifier(loss='squared_hinge'),
                    # 'Perceptron': SGDClassifier(loss='perceptron'))
                    }

        learners_params = {
            'SGDClassifier': [{'loss': ['hinge'], 'penalty': ['l2', 'l1'], 'alpha': [0.001, 0.0001]},
                              {'loss': ['perceptron'], 'penalty': ['l2', 'l1'], 'alpha': [0.001, 0.0001]},
                              {'loss': ['huber'], 'penalty': ['l2', 'l1'], 'alpha': [0.001, 0.0001]},
                              {'loss': ['modified_huber'], 'penalty': ['l2', 'l1'],
                               'alpha': [0.001, 0.0001]}],
            'Passive-Aggressive': [
                {'loss': ['squared_hinge'], 'C': [0.8, 1.0, 1.2], 'fit_intercept': [True, False]}],
            'LinearSVC': [{'loss': ['hinge', 'squared_hinge'], 'dual': [True, False]}]}

        reports_1m = {}
        reports_broad_twitter_corpus = {}
        # self.cross_val(learners_params['LinearSVC'], learners['LinearSVC'])

        for learner_name, learner in learners.items():
            print("Run " + learner_name)
            reports_1m[str(learner_name) + '0'] = self.preprocess_and_fit(self.df_1m, learner, False)
            reports_1m[str(learner_name) + '1'] = self.preprocess_and_fit(self.df_1m_brown, learner, False)
            reports_broad_twitter_corpus[str(learner_name) + '0'] = \
                self.preprocess_and_fit(self.df_broad_twitter_corpus, learner)
            reports_broad_twitter_corpus[str(learner_name) + '1'] = \
                self.preprocess_and_fit(self.df_broad_twitter_corpus_brown, learner)

        for report_name, report in reports_1m.items():
            print("\n" + report_name + ": \n" + str(report))
        print('\n### Broad Twitter Corpus: ###')
        for report_name, report in reports_broad_twitter_corpus.items():
            print("\n" + report_name + ": \n" + str(report['micro avg']))

    def score(self, estimator, X_test, y_test):
        '''
        Calculate F1 score on the given estimator and test data
        '''
        labels = list(np.unique(y_test))
        labels.pop()
        return f1_score(y_true=y_test,
                        y_pred=estimator.predict(X_test),
                        labels=labels,
                        average='micro')

    def run_learners_graphs(self, first_df, second_df, data_name):
        '''
        Creates different graphs in order to compare results
        '''
        learner = LinearSVC(dual=True, loss='hinge')

        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=73)
        # cv = KFold(n_splits=5, random_state=73, shuffle=True)
        frac_sizes = np.linspace(.1, 1.0, 5)
        X1, y1 = self.encode_and_return_X_y(first_df)
        X2, y2 = self.encode_and_return_X_y(second_df)

        fit_times_mean, fit_times_std, test_scores_mean, test_scores_std, \
        train_scores_mean, train_scores_std, train_sizes = \
            self.learning_curve_plot(X1, cv, learner, frac_sizes, y1)

        fit_times_mean2, fit_times_std2, test_scores_mean2, test_scores_std2, \
        train_scores_mean2, train_scores_std2, train_sizes2 = \
            self.learning_curve_plot(X2, cv, learner, frac_sizes, y2)

        # Plot learning curve
        self.plot_graphs(
            fit_times_mean, fit_times_std, test_scores_mean, test_scores_std, train_scores_mean,
            train_scores_std, train_sizes,
            fit_times_mean2, fit_times_std2, test_scores_mean2, test_scores_std2, train_scores_mean2,
            train_scores_std2, train_sizes2,
            data_name)

    def learning_curve_plot(self, X1, cv, learner, train_sizes, y1):
        '''
        Creates learning curve graphs
        '''
        train_sizes, train_scores, test_scores, fit_times, _ = \
            learning_curve(learner, X1, y1, cv=cv,
                           train_sizes=train_sizes,
                           return_times=True,
                           scoring=self.score)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)
        return fit_times_mean, fit_times_std, test_scores_mean, test_scores_std, train_scores_mean, train_scores_std, train_sizes

    def plot_graphs(self,
                    fit_times_mean, fit_times_std, test_scores_mean, test_scores_std, train_scores_mean,
                    train_scores_std, train_sizes,
                    fit_times_mean2, fit_times_std2, test_scores_mean2, test_scores_std2, train_scores_mean2,
                    train_scores_std2, train_sizes2,
                    graph_name):
        # Plot n_samples vs fit_times
        plt.grid()
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color="c")
        plt.fill_between(train_sizes2, train_scores_mean2 - train_scores_std2,
                         train_scores_mean2 + train_scores_std2, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="b")
        plt.fill_between(train_sizes2, test_scores_mean2 - test_scores_std2,
                         test_scores_mean2 + test_scores_std2, alpha=0.1, color="m")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="c", label="Training score")
        plt.plot(train_sizes2, train_scores_mean2, 'o-', color="r",
                 label="Training score with Brown Clusters")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="b", label="Cross-validation score")
        plt.plot(train_sizes2, test_scores_mean2, 'o-', color="m",
                 label="Cross-validation score with Brown Clusters")
        plt.xlabel("Training examples")
        plt.ylabel("F1 score")
        plt.title(graph_name + " - Learning curve by examples")
        plt.legend(loc="best")
        plt.savefig('Score_' + graph_name)
        plt.close()

        plt.grid()
        plt.plot(train_sizes, fit_times_mean, 'o-', label="Without Brown Clusters")
        plt.plot(train_sizes2, fit_times_mean2, 'o-', label="With Brown Clusters")
        plt.fill_between(train_sizes, fit_times_mean - fit_times_std, fit_times_mean + fit_times_std,
                         alpha=0.1)
        plt.fill_between(train_sizes2, fit_times_mean2 - fit_times_std2, fit_times_mean2 + fit_times_std2,
                         alpha=0.1)
        plt.xlabel("Training examples")
        plt.ylabel("Fit times")
        plt.title(graph_name + " - Scalability of the model")
        plt.legend(loc="best")
        plt.savefig('Scalability_' + graph_name)
        plt.close()

        # Plot fit_time vs score
        plt.grid()
        plt.plot(fit_times_mean, test_scores_mean, 'o-', label='Without Brown Clusters')
        plt.plot(fit_times_mean2, test_scores_mean2, 'o-', label='With Brown Clusters')
        plt.fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
        plt.fill_between(fit_times_mean2, test_scores_mean2 - test_scores_std2,
                         test_scores_mean2 + test_scores_std2, alpha=0.1)
        plt.xlabel("Fit times")
        plt.ylabel("F1 Score")
        plt.title(graph_name + " - Learning curve by fit time")
        plt.legend(loc="best")
        plt.savefig('Performance_' + graph_name)
        plt.close()

    def cross_val(self, learner_params, learner):
        '''
        Run cross validation on the given learner in order to find the best parameters from learner_params
        '''
        X, y = self.encode_and_return_X_y(self.df_1m_brown)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.TEST_PER, random_state=13)

        scores = ['precision', 'recall']
        for score in scores:
            print("# Tuning hyper-parameters for %s" % score + "\n")
            clf = GridSearchCV(learner, learner_params, cv=5)
            clf.fit(X_train, y_train)
            print("Best parameters set found on development set:\n")
            print(clf.best_params_ + "\n")
            print("Grid scores on development set:\n")
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    def run_learner_accuracy_graphs(self):
        learner.run_learner()
        learner.run_learners_graphs(self.df_1m, self.df_1m_brown, '1M')
        learner.run_learners_graphs(self.df_broad_twitter_corpus,
                                    self.df_broad_twitter_corpus_brown,
                                    'Broad Twitter Corpus')


if __name__ == "__main__":
    learner = NERLearner()
    learner.run_learner_accuracy_graphs()
