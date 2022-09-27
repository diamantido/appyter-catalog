# Adapted from code written by Daniel Clarke

import pandas as pd
from tqdm import tqdm
import numpy as np
import datetime

'''
Sources are currently human, mouse, and rat, in order of descending priority.
'''
sources = [
    'ftp://ftp.ncbi.nih.gov/gene/DATA/GENE_INFO/Mammalia/Rattus_norvegicus.gene_info.gz',
    'ftp://ftp.ncbi.nih.gov/gene/DATA/GENE_INFO/Mammalia/Mus_musculus.gene_info.gz',
    'ftp://ftp.ncbi.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz'
]


def get_dictionary(path, mapfrom):
    '''
    Returns two dictionaries, the first a mapping from symbols to approved gene
    symbols (synonyms), the second a mapping from approved symbols to Entrez
    Gene IDs from an NCBI gene info source designated by path.
    '''
    column = {
        'synonyms': 'Synonyms',
        'ensembl': 'dbXrefs'
    }[mapfrom]
    ncbi = pd.read_csv(path, sep='\t', usecols=[
        'Symbol', 'GeneID', column])

    def split_list(v): return v.split('|') if type(v) == str else []
    ncbi[column] = ncbi[column].apply(split_list)

    # Map existing entities to NCBI Genes
    symbol_lookup = {}
    geneid_lookup = {}
    for i in ncbi.index:
        approved_sym = ncbi.loc[i, 'Symbol']
        v = ncbi.loc[i, 'GeneID']
        geneid_lookup[approved_sym] = v if v != '-' else np.nan

        if mapfrom == 'synonyms':
            for sym in [approved_sym] + ncbi.loc[i, column]:
                if not (sym == '-'):
                    symbol_lookup[sym] = approved_sym
        elif mapfrom == 'ensembl':
            for sym in ncbi.loc[i, column]:
                if sym.startswith('Ensembl:'):
                    symbol_lookup[sym[len('Ensembl:'):]] = approved_sym

    return symbol_lookup, geneid_lookup


def get_lookups(mapfrom='synonyms'):
    '''
    Returns two dictionaries, the first a mapping from symbols to approved gene
    symbols (synonyms), the second a mapping from approved symbols to Entrez
    Gene IDs.

    '''
    symbol_lookup = {}
    geneid_lookup = {}

    for source in tqdm(sources, desc='Gathering sources'):
        sym, gene = get_dictionary(source, mapfrom)
        symbol_lookup.update(sym)
        geneid_lookup.update(gene)

    return symbol_lookup, geneid_lookup


def save_lookup(symbol_lookup, geneid_lookup):
    '''
    Save the lookups as pandas DataFrames. They are saved as:
        symbol_lookup_<year>_<month>.csv
        geneid_lookup_<year>_<month>.csv
    '''
    date = str(datetime.date.today())[0:7].replace('-', '_')
    symbol = pd.DataFrame.from_dict(
        symbol_lookup, orient='index', columns=['Approved Symbol'])
    geneid = pd.DataFrame.from_dict(
        geneid_lookup, orient='index', columns=['Entrez Gene ID'])

    symbol.to_csv('symbol_lookup_{}.csv'.format(date), sep='\t')
    geneid.to_csv('geneid_lookup_{}.csv'.format(date), sep='\t')


def load_lookup(symbol_path, geneid_path):
    '''
    Loads the lookups from custom paths. The files should be tab-separated 
    files. Returns the symbol and geneid lookups as dictionaries.
    '''
    symbol = pd.read_csv(symbol_path, sep='\t', na_filter=False)
    geneid = pd.read_csv(geneid_path, sep='\t', na_filter=False)

    symbol_lookup = dict(zip(symbol.iloc[:, 0], symbol.iloc[:, 1]))
    geneid_lookup = dict(zip(geneid.iloc[:, 0], geneid.iloc[:, 1]))

    return symbol_lookup, geneid_lookup
