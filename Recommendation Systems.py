
"""LIBRARIES"""
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


                              """1-USER BASED RECEOMMENDATION SYSTEMS"""
def normalize(pred_ratings):
    '''
    This function will normalize the input pred_ratings

    params:
        pred_ratings (List -> List) : The prediction ratings
    '''
    return (pred_ratings - pred_ratings.min()) / (pred_ratings.max() - pred_ratings.min())

def generate_prediction_df(mat, pt_df, n_factors):
    '''
    This function will calculate the single value decomposition of the input matrix
    given n_factors. It will then generate and normalize the user rating predictions.

    params:
        mat (CSR Matrix) : scipy csr matrix corresponding to the pivot table (pt_df)
        pt_df (DataFrame) : pandas dataframe which is a pivot table
        n_factors (Integer) : Number of singular values and vectors to compute.
                              Must be 1 <= n_factors < min(mat.shape).
    '''

    if not 1 <= n_factors < min(mat.shape):
        raise ValueError("Must be 1 <= n_factors < min(mat.shape)")

    # matrix factorization
    u, s, v = svds(mat, k = n_factors)
    s = np.diag(s)

    # calculate pred ratings
    pred_ratings = np.dot(np.dot(u, s), v)
    pred_ratings = normalize(pred_ratings)

    # convert to df
    pred_df = pd.DataFrame(
        pred_ratings,
        columns = pt_df.columns,
        index = list(pt_df.index)
    ).transpose()
    return pred_df

def recommend_items(pred_df, usr_id, n_recs):
    '''
    Given a usr_id and pred_df this function will recommend
    items to the user.

    params:
        pred_df (DataFrame) : generated from `generate_prediction_df` function
        usr_id (Integer) : The user you wish to get item recommendations for
        n_recs (Integer) : The number of recommendations you want for this user
    '''

    usr_pred = pred_df[usr_id].sort_values(ascending = False).reset_index().rename(columns = {usr_id : 'sim'})
    rec_df = usr_pred.sort_values(by = 'sim', ascending = False).head(n_recs)
    return rec_df

if __name__ == '__main__':
    # constants
    PATH = '../data/data.csv'

    # import data
    df = pd.read_csv(PATH)
    print(df.shape)

    # generate a pivot table with readers on the index and books on the column and values being the ratings
    pt_df = df.pivot_table(
        columns = 'book_id',
        index = 'reader_id',
        values = 'book_rating'
    ).fillna(0)

    # convert to a csr matrix
    mat = pt_df.values
    mat = csr_matrix(mat)

    pred_df = generate_prediction_df(mat, pt_df, 10)

    # generate recommendations
    print(recommend_items(pred_df, 5, 5))



                            """2-CONTENT BASED RECEOMMENDATION SYSTEMS"""
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm


def normalize(data):
    '''
    This function will normalize the input data to be between 0 and 1

    params:
        data (List) : The list of values you want to normalize

    returns:
        The input data normalized between 0 and 1
    '''
    min_val = min(data)
    if min_val < 0:
        data = [x + abs(min_val) for x in data]
    max_val = max(data)
    return [x / max_val for x in data]


def ohe(df, enc_col):
    '''
    This function will one hot encode the specified column and add it back
    onto the input dataframe

    params:
        df (DataFrame) : The dataframe you wish for the results to be appended to
        enc_col (String) : The column you want to OHE

    returns:
        The OHE columns added onto the input dataframe
    '''

    ohe_df = pd.get_dummies(df[enc_col])
    ohe_df.reset_index(drop=True, inplace=True)
    return pd.concat([df, ohe_df], axis=1)


class CBRecommend():
    def __init__(self, df):
        self.df = df

    def cosine_sim(self, v1, v2):
        '''
        This function will calculate the cosine similarity between two vectors
        '''
        return sum(dot(v1, v2) / (norm(v1) * norm(v2)))

    def recommend(self, book_id, n_rec):
        """
        df (dataframe): The dataframe
        song_id (string): Representing the song name
        n_rec (int): amount of rec user wants
        """

        # calculate similarity of input book_id vector w.r.t all other vectors
        inputVec = self.df.loc[book_id].values
        self.df['sim'] = self.df.apply(lambda x: self.cosine_sim(inputVec, x.values), axis=1)

        # returns top n user specified books
        return self.df.nlargest(columns='sim', n=n_rec)


if __name__ == '__main__':
    # constants
    PATH = '../data/data.csv'

    # import data
    df = pd.read_csv(PATH)

    # normalize the num_pages, ratings, price columns
    df['num_pages_norm'] = normalize(df['num_pages'].values)
    df['book_rating_norm'] = normalize(df['book_rating'].values)
    df['book_price_norm'] = normalize(df['book_price'].values)

    # OHE on publish_year and genre
    df = ohe(df=df, enc_col='publish_year')
    df = ohe(df=df, enc_col='book_genre')
    df = ohe(df=df, enc_col='text_lang')

    # drop redundant columns
    cols = ['publish_year', 'book_genre', 'num_pages', 'book_rating', 'book_price', 'text_lang']
    df.drop(columns=cols, inplace=True)
    df.set_index('book_id', inplace=True)

    # ran on a sample as an example
    t = df.copy()
    cbr = CBRecommend(df=t)
    print(cbr.recommend(book_id=t.index[0], n_rec=5))


                              """3-HYBRID RECOMMENDATION SYSTEMS"""
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Reader, Dataset, accuracy
from surprise.model_selection import train_test_split


def hybrid(reader_id, book_id, n_recs, df, cosine_sim, svd_model):
    '''
    This function represents a hybrid recommendation system, it will have the following flow:
        1. Use a content-based model (cosine_similarity) to compute the 50 most similar books
        2. Compute the predicted ratings that the user might give these 50 books using a collaborative
           filtering model (SVD)
        3. Return the top n books with the highest predicted rating

    params:
        reader_id (Integer) : The reader_id
        book_id (Integer) : The book_id
        n_recs (Integer) : The number of recommendations you want
        df (DataFrame) : Original dataframe with all book information
        cosine_sim (DataFrame) : The cosine similarity dataframe
        svd_model (Model) : SVD model
    '''

    # sort similarity values in decreasing order and take top 50 results
    sim = list(enumerate(cosine_sim[int(book_id)]))
    sim = sorted(sim, key=lambda x: x[1], reverse=True)
    sim = sim[1:50]

    # get book metadata
    book_idx = [i[0] for i in sim]
    books = df.iloc[book_idx][['book_id', 'book_rating', 'num_pages', 'publish_year', 'book_price', 'reader_id']]

    # predict using the svd_model
    books['est'] = books.apply(lambda x: svd_model.predict(reader_id, x['book_id'], x['book_rating']).est, axis=1)

    # sort predictions in decreasing order and return top n_recs
    books = books.sort_values('est', ascending=False)
    return books.head(n_recs)


if __name__ == '__main__':
    # constants
    PATH = '../data/data.csv'

    # import data
    df = pd.read_csv(PATH)

    # content based
    rmat = df.pivot_table(
        columns='book_id',
        index='reader_id',
        values='book_rating'
    ).fillna(0)

    # Compute the cosine similarity matrix
    cosine_sim = cosine_similarity(rmat, rmat)
    cosine_sim = pd.DataFrame(cosine_sim, index=rmat.index, columns=rmat.index)

    # collaborative filtering
    reader = Reader()
    data = Dataset.load_from_df(df[['reader_id', 'book_id', 'book_rating']], reader)

    # split data into train test
    trainset, testset = train_test_split(data, test_size=0.3, random_state=10)

    # train
    svd = SVD()
    svd.fit(trainset)

    # run the trained model against the testset
    test_pred = svd.test(testset)

    # get RMSE
    accuracy.rmse(test_pred, verbose=True)

    # generate recommendations
    r_id = df['reader_id'].values[0]
    b_id = df['book_id'].values[0]
    n_recs = 5
    print(hybrid(r_id, b_id, n_recs, df, cosine_sim, svd))
