# Adapted from code created by Moshe Silverstein and Charles Dai

import datetime
import os
import zipfile

import numpy as np
import pandas as pd
import scipy.spatial.distance as dist
import scipy.sparse as sp
from statsmodels.distributions.empirical_distribution import ECDF
from tqdm import tqdm


def remove_impute(df):
    '''
    Removes rows and columns that have more than 95% of their data missing,
    or 0. Replacing any missing data leftover after removal with
    the means of the rows.
    '''
    r, c = df.shape
    df.loc[np.sum(np.logical_or(np.isnan(df), df == 0), axis=1) < 0.05 * r,
           np.sum(np.logical_or(np.isnan(df), df == 0), axis=0) < 0.05 * c]

    return df.fillna(df.mean(axis=0))


def merge(df, axis):
    '''
    Merges duplicate rows or columns, depending on the axis specified. The
    final values of the merged rows or columns is determined by the method.
    '''
    if axis == 'column':
        return df.groupby(df.columns, axis=1).mean()
    elif axis == 'row':
        return df.groupby(level=0, axis=0).mean()


def quantile_normalize(df):
    '''
    Performs quantile normalization on the input DataFrame.
    '''
    # from ayhan on StackOverflow
    rank_mean = df.stack().groupby(
        df.rank(method='first').stack().astype(int)).mean()
    return df.rank(method='min').stack().astype(int).map(
        rank_mean).unstack()


def zscore(df, epsilon=0):
    '''
    Calculates the modified z-score of df according to the specified axis.

    Parameters:
        axis - the axis on which to calculate the z-scores. Either 'row' or 'column'
        epsilon - small adjustment in the case of divide by 0 errors.
    '''
    np.seterr(divide='ignore', invalid='ignore')
    median_y = np.median(df, axis=1)[:, np.newaxis]
    abs_dev = np.abs(df - median_y)
    median_dev = np.median(abs_dev, axis=1)
    mean_dev = np.mean(abs_dev, axis=1)
    median_abs_dev = np.broadcast_to(median_dev[:, np.newaxis], df.shape)
    mean_abs_dev = np.broadcast_to(mean_dev[:, np.newaxis], df.shape)
    modified_z_scores = np.where(median_abs_dev != 0,
                                 0.6745 * (df - median_y) / median_abs_dev,
                                 (df - median_y) / (1.253314 * mean_abs_dev + epsilon))

    return pd.DataFrame(data=modified_z_scores, index=df.index,
                        columns=df.columns)


def log2(df):
    '''
    Returns a dataframe with the adjusted log2 values of the input.
    '''
    return np.log2(df + 1)


def map_symbols(df, symbol_lookup, remove_duplicates=False):
    '''
    Replaces the index of the df, which are gene names, with
    corresponding approved gene symbols according to the given symbol_lookup 
    dictionary. If any gene names are not in the mapping, they are discarded 
    from the DataFrame.
    '''
    tqdm.pandas()
    df = df.reset_index()

    df.iloc[:, 0] = df.iloc[:, 0].progress_map(
        lambda x: symbol_lookup.get(x, np.nan))

    df = df.dropna(subset=[df.columns[0]])
    if remove_duplicates:
        df = df.drop_duplicates()
    df = df.set_index(df.columns[0])
    return df


def binary_matrix(df):
    '''
    Creates an adjacency matrix from df, which is a gene-attribute edge
    list.
    '''
    matrix = pd.crosstab(df.index, df.iloc[:, 0]) > 0
    matrix.index.name = df.index.name
    matrix.columns.name = df.columns[0]
    return matrix


def ternary_matrix(df):
    '''
    Returns the input matrix with all significant values, greater than 0.95
    or less than -0.95, mapped to 1 or -1, respectively. All other values
    are mapped to 0.
    '''
    def mapter(x):
        if x >= 0.95:
            return 1
        elif x <= -0.95:
            return -1
        else:
            return 0

    return df.applymap(mapter)


def save_setlib(df, lib, direction, path, name):
    '''
    If lib = 'gene', this creates a file which lists all attributes and the
    genes that are correlated in the direction given with that attribute.

    If lib = 'attribute', this creates a file which lists all genes and the
    attributes that are correlated in the direction given with that gene.
    The year and month are added at the end of the name. The path the file is
    saved to is thus
        path + name + '_<year>_<month>.gmt'
    '''
    filenameGMT = file_name(path, name, 'gmt')
    direction = {'up': 1, 'down': -1}[direction]

    if not (lib == 'gene' or lib == 'attribute'):
        return
    if lib == 'attribute':
        df = df.T

    with open(filenameGMT, 'w') as f:
        arr = df.reset_index(drop=True).to_numpy(dtype=np.int_)
        attributes = df.columns

        w, h = arr.shape
        for i in tqdm(range(h)):
            print(attributes[i], *df.index[arr[:, i] == direction],
                  sep='\t', end='\n', file=f)


def similarity_matrix(df, metric, dtype=None, sparse=False):
    '''
    Creates a similarity matrix between the rows of the df based on
    the metric specified. The resulting matrix has both rows and columns labeled
    by the index of df.
    '''
    if sparse and metric == 'jaccard':
        # from na-o-ys on Github
        sparse = sp.csr_matrix(df.to_numpy(dtype=np.bool).astype(np.int))
        cols_sum = sparse.getnnz(axis=1)
        ab = sparse * sparse.T
        denom = np.repeat(cols_sum, ab.getnnz(axis=1)) + \
            cols_sum[ab.indices] - ab.data
        ab.data = ab.data / denom
        similarity_matrix = ab.todense()
        np.fill_diagonal(similarity_matrix, 1)

    else:
        similarity_matrix = dist.pdist(df.to_numpy(dtype=dtype), metric)
        similarity_matrix = dist.squareform(similarity_matrix)
        similarity_matrix = 1 - similarity_matrix

    similarity_df = pd.DataFrame(
        data=similarity_matrix, index=df.index, columns=df.index)
    similarity_df.index.name = None
    similarity_df.columns.name = None
    return similarity_df


def gene_list(df, geneid_lookup):
    '''
    Creates a list of genes and the corresponding Entrez Gene IDs(supplied by
    the NCBI)

    Note: this differs from the previous function in its behavior with dealing
    with genes that do not have an ID. This function will set the id of the gene
    to -1, whereas the previous script will set them to np.nan.
    '''
    gene_ids = np.array([geneid_lookup.get(x, -1)
                         if np.isfinite(geneid_lookup.get(x, -1))
                         else -1 for x in tqdm(df.index)], dtype=np.int_)
    df = pd.DataFrame(gene_ids, index=df.index, columns=['Gene ID'])
    return df


def attribute_list(df, metaData=None):
    '''
    Creates a list of attributes in the form of a DataFrame, with the attributes
    as the indices. If metaData is specified, it returns appends the attributes
    of df onto the metaData DataFrame.
    '''
    if metaData is not None:
        attribute_list = metaData.reindex(df.columns)
        attribute_list.index.name = df.columns.name
    else:
        attribute_list = pd.DataFrame(index=df.columns)
    return attribute_list


def standardized_matrix(df):
    '''
    Creates a standardized matrix by using an emperical CDF for each row.
    Each row in the df should represent a single gene.

    Requires:
    Indices of the DataFrame are unique.
    '''
    arr = df.to_numpy(copy=True)

    def process(array):
        ourECDF = ECDF(array)
        array = ourECDF(array)
        mean = np.mean(array)
        array = 2 * (array - mean)
        return array

    for i in tqdm(range(arr.shape[0])):
        arr[i, :] = process(arr[i, :])

    values = arr.flatten()
    ourECDF = ECDF(values)
    ourECDF = ourECDF(values).reshape(arr.shape)

    mean = np.mean(ourECDF)
    ourECDF = 2 * (ourECDF - mean)
    newDF = pd.DataFrame(data=ourECDF, index=df.index,
                         columns=df.columns)
    return newDF


def edge_list(df):
    '''
    Creates the gene-attribute edge list from the given input DataFrame,
    attribute and gene lists. The year and month are added at the
    end of the name. The path the file is saved to is thus
        path + name + '_<year>_<month>.gmt'
    Also prints the number of cells in df that are statistically
    significant, i.e. > 0.95 confidence.
    Requires:
        attributelist and genelist were generated from running
        createAttributeList and createGeneList on df, respectively.
    '''
    count = np.sum(np.sum(df >= 0.95) + np.sum(df <= -0.95))
    df = df.stack()
    df.name = 'Weight'
    print('The number of statisticaly relevant gene-attribute associations is: %d' % count)
    return df


def file_name(path, name, ext):
    '''
    Returns the file name by taking the path and name, adding the year and month
    and then the extension. The final string returned is thus
        '<path>/<name>_<year>_<month>.ext'
    '''
    date = str(datetime.date.today())[0:7].replace('-', '_')
    filename = ''.join([name, '_', date, '.', ext])
    return os.path.join(path, filename)


def save_data(df, path, name, compression=None, ext='tsv',
              symmetric=False, dtype=None, **kwargs):
    '''
    Save df according to the compression method given. 
    compression can take these values:
        None or 'gmt' - defaults to pandas to_csv() function.
        'gzip' - uses the gzip compression method of the pandas to_csv() function
        'npz' - converts the DataFrame to a numpy array, and saves the array.
                The array is stored as 'axes[0]_axes[1]'. If symmetric is true,
                it is stored as 'axes[0]_axes[1]_symmetric' instead.
    ext is only used if compression is None or 'gzip'. The extension of the file
    will be .ext, or .ext.gz if 'gzip' is specified.
    axes must only be specified if compression is 'npz'. It is a string tuple
    that describes the index and columns df, i.e. (x, y) where x, y = 
    'gene' or 'attribute'.
    symmetric is only used if compression is 'npz', and indicates if df
    is symmetric and can be stored as such. 
    dtype is only used if compression is 'npz', and indicates a dtype that the
    array can be cast to before storing.

    The year and month are added at the end of the name. The path the file is 
    saved to is thus
        path + name + '_<year>_<month>.ext'
    where ext is .ext, .ext.gz, or .npz depending on the compression method.
    '''

    if compression is None:
        name = file_name(path, name, ext)
        df.to_csv(name, sep='\t', **kwargs)
    elif compression == 'gzip':
        name = file_name(path, name, ext + '.gz')
        df.to_csv(name, sep='\t', compression='gzip', **kwargs)
    elif compression == 'npz':
        name = file_name(path, name, 'npz')

        data = df.to_numpy(dtype=dtype)
        index = np.array(df.index)
        columns = np.array(df.columns)

        if symmetric:
            data = np.triu(data)
            np.savez_compressed(name, symmetric=data, index=index)
        else:
            np.savez_compressed(name, nonsymmetric=data,
                                index=index, columns=columns)


def load_data(filename):
    '''
    Loads a pandas DataFrame stored in a .npz data numpy array format.
    '''
    with np.load(filename, allow_pickle=True) as data_load:
        arrays = data_load.files
        if arrays[0] == 'symmetric':
            data = data_load['symmetric']
            index = data_load['index']
            data = data + data.T - np.diag(data.diagonal())
            df = pd.DataFrame(data=data, index=index, columns=index)
            return df
        elif arrays[0] == 'nonsymmetric':
            data = data_load['nonsymmetric']
            index = data_load['index']
            columns = data_load['columns']
            df = pd.DataFrame(data=data, index=index, columns=columns)
            return df


def archive(path, output_name=''):
    with zipfile.ZipFile(output_name+'output_archive.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(path):
            for f in files:
                zipf.write(os.path.join(root, f))
