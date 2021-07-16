import os
import pandas as pd
import numpy as np


class ClusterMaker:

    TWITTER_FILE_NAME = os.path.join("Data", "twitter_50mpaths2")
    TWITTER_FILE_NAME_SMALL = os.path.join("Data", "twitter_1kpaths")

    def load_twitter_clusters(self, file_name):
        '''
        Saving it csv file of the twitter brown clusters
        '''
        self.twitter_brown_clusters = pd.read_csv(file_name,
                                                  sep='\t',
                                                  encoding="ISO-8859-1",
                                                  names=['Cluster', 'Word', 'Frequency'],
                                                  dtype={'Cluster': object, 'Word': object,
                                                         'Frequency': np.uint8})

    def add_brown_clusters_as_feature(self, df: pd.DataFrame, fill_nan_zero=False) -> pd.DataFrame:
        '''
        Adds the cluster numbers to a df_to_add_features
        Assumes that a column named 'word' is found and is the key for the merge
        :param fill_nan_zero:
        :param df: DataFrame to add column (feature) to
        :return: DataFrame with the added column
        '''
        if self.twitter_brown_clusters is None:
            self.load_twitter_clusters(self.TWITTER_FILE_NAME)
        # df['Capitalize'] = np.where(df.Word.str.capitalize()==df.Word, 1, 0)

        word_clusters = self.twitter_brown_clusters.drop('Frequency', 1)
        new_word = df.copy()
        lower_case_word = new_word.Word.str.lower()

        new_word['Word'] = lower_case_word
        new_word = pd.merge(new_word, word_clusters, how='left', on=['Word'])
        new_word['Word'] = df['Word']

        if fill_nan_zero:
            new_word['Cluster'] = new_word['Cluster'].fillna(value='0')

        return new_word

    def set_same_cluster_same_tokens(self, df):
        '''
        For each word without a cluster in :param:df, adds a new cluster that only that token is in it.
        '''
        all_nan = df.iloc[df['Cluster'].isnull().values]['Word']
        new_clusters = dict()
        cluster_generator = iter(self.binary_generator())
        for nan in np.unique(all_nan):
            new_clusters[nan] = next(cluster_generator)
        new_dict = dict()
        for index in all_nan.to_dict():
            new_dict[index] = new_clusters[all_nan[index]]
        new_cluster_col = df['Cluster'].fillna(new_dict)
        df['Cluster'] = new_cluster_col
        return df

    def binary_generator(self):
        '''
        Generator function that return a new binary number, starting from 1001 and increasing.
        '''
        i = 1001
        while True:
            yield f'{i:#b}'[2:]
            i = i+1


if __name__ == '__main__':
    twitter = ClusterMaker()
