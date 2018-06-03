# nice method
import numpy as np
import pandas as pd
import lightgbm
import time
from gensim.models.Word2Vec
# from joblib import Parallel, delayed
import multiprocessing
from datetime import datetime
pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)

def f1_score_single(y_true, y_pred):
    y_true = set(y_true)
    y_pred = set(y_pred)
    cross_size = len(y_true & y_pred)
    if cross_size == 0: return 0.
    p = 1. * cross_size / len(y_pred)
    r = 1. * cross_size / len(y_true)
    return 2 * p * r / (p + r)


def f1_score_single_one_para((y_true, y_pred)):
    y_true = set(y_true)
    y_pred = set(y_pred)
    cross_size = len(y_true & y_pred)
    if cross_size == 0: return 0.
    p = 1. * cross_size / len(y_pred)
    r = 1. * cross_size / len(y_true)
    return 2 * p * r / (p + r)


def f1_score(y_true, y_pred):
    return np.mean([f1_score_single(x, y) for x, y in zip(y_true, y_pred)])


import numpy as np
import itertools


def fast_search(prob, dtype=np.float32):
    size = len(prob)
    fk = np.zeros((size + 1), dtype=dtype)
    C = np.zeros((size + 1, size + 1), dtype=dtype)
    S = np.empty((2 * size + 1), dtype=dtype)
    S[:] = np.nan
    for k in range(1, 2 * size + 1):
        S[k] = 1. / k
    roots = (prob - 1.0) / prob
    for k in range(size, 0, -1):
        poly = np.poly1d(roots[0:k], True)
        factor = np.multiply.reduce(prob[0:k])
        C[k, 0:k + 1] = poly.coeffs[::-1] * factor
        for k1 in range(size + 1):
            fk[k] += (1. + 1.) * k1 * C[k, k1] * S[k + k1]
        for i in range(1, 2 * (k - 1)):
            S[i] = (1. - prob[k - 1]) * S[i] + prob[k - 1] * S[i + 1]

    return fk


# a = map(f1_score_single_one_para, [([1, 1], [1, 0]), ([1, 1], [1, 1]), ([1, 1], [1, 0])])
# print np.mean(a)

def load_data(path_data):
    '''
    --------------------------------order_product--------------------------------
    * Unique in order_id + product_id
    '''
    priors = pd.read_csv(path_data + 'order_products__prior.csv',
                         dtype={
                             'order_id': np.int32,
                             'product_id': np.uint16,
                             'add_to_cart_order': np.int16,
                             'reordered': np.int8})
    train = pd.read_csv(path_data + 'order_products__train.csv',
                        dtype={
                            'order_id': np.int32,
                            'product_id': np.uint16,
                            'add_to_cart_order': np.int16,
                            'reordered': np.int8})
    '''
    --------------------------------order--------------------------------
    * This file tells us which set (prior, train, test) an order belongs
    * Unique in order_id
    * order_id in train, prior, test has no intersection
    * this is the #order_number order of this user
    '''
    orders = pd.read_csv(path_data + 'orders.csv',
                         dtype={
                             'order_id': np.int32,
                             'user_id': np.int64,
                             'eval_set': 'category',
                             'order_number': np.int16,
                             'order_dow': np.int8,
                             'order_hour_of_day': np.int8,
                             'days_since_prior_order': np.float32})

    #  order in prior, train, test has no duplicate
    #  order_ids_pri = priors.order_id.unique()
    #  order_ids_trn = train.order_id.unique()
    #  order_ids_tst = orders[orders.eval_set == 'test']['order_id'].unique()
    #  print(set(order_ids_pri).intersection(set(order_ids_trn)))
    #  print(set(order_ids_pri).intersection(set(order_ids_tst)))
    #  print(set(order_ids_trn).intersection(set(order_ids_tst)))

    '''
    --------------------------------product--------------------------------
    * Unique in product_id
    '''
    products = pd.read_csv(path_data + 'products.csv')
    aisles = pd.read_csv(path_data + "aisles.csv")
    departments = pd.read_csv(path_data + "departments.csv")
    sample_submission = pd.read_csv(path_data + "sample_submission.csv")
    order_streaks = pd.read_csv(path_data + "order_streaks.csv")

    return priors, train, orders, products, aisles, departments, sample_submission, order_streaks


class tick_tock:
    def __init__(self, process_name, verbose=1):
        self.process_name = process_name
        self.verbose = verbose

    def __enter__(self):
        if self.verbose:
            print(self.process_name + " begin ......")
            self.begin_time = time.time()

    def __exit__(self, type, value, traceback):
        if self.verbose:
            end_time = time.time()
            print(self.process_name + " end ......")
            print('time lapsing {0} s \n'.format(end_time - self.begin_time))


def ka_add_groupby_features_1_vs_n(df, group_columns_list, agg_dict, only_new_feature=True):
    '''Create statistical columns, group by [N columns] and compute stats on [N column]

       Parameters
       ----------
       df: pandas dataframe
          Features matrix
       group_columns_list: list_like
          List of columns you want to group with, could be multiple columns
       agg_dict: python dictionary

       Return
       ------
       new pandas dataframe with original columns and new added columns

       Example
       -------
       {real_column_name: {your_specified_new_column_name : method}}
       agg_dict = {'user_id':{'prod_tot_cnts':'count'},
                   'reordered':{'reorder_tot_cnts_of_this_prod':'sum'},
                   'user_buy_product_times': {'prod_order_once':lambda x: sum(x==1),
                                              'prod_order_more_than_once':lambda x: sum(x==2)}}
       ka_add_stats_features_1_vs_n(train, ['product_id'], agg_dict)
    '''
    with tick_tock("add stats features"):
        try:
            if type(group_columns_list) == list:
                pass
            else:
                raise TypeError(group_columns_list + "should be a list")
        except TypeError as e:
            print(e)
            raise

        df_new = df.copy()
        grouped = df_new.groupby(group_columns_list)

        the_stats = grouped.agg(agg_dict)
        the_stats.columns = the_stats.columns.droplevel(0)
        the_stats.reset_index(inplace=True)
        if only_new_feature:
            df_new = the_stats
        else:
            df_new = pd.merge(left=df_new, right=the_stats, on=group_columns_list, how='left')

    return df_new


def get_rolling_target(df, group_column, target_column_list, prefix="hehe", drop_raw_col=False):
    df_old = df.copy()
    grouped = df_old.groupby(group_column)
    rs = []
    for target_column in target_column_list:
        a = [grouped[target_column].rolling(5).fillna(0).mean().shift(-5),
             grouped[target_column].rolling(5).fillna(0).std().shift(-5),
             grouped[target_column].rolling(5).fillna(0).sum().shift(-5),
             grouped[target_column].rolling(5).fillna(0).mean().shift(),
             grouped[target_column].rolling(5).fillna(0).std().shift(),
             grouped[target_column].rolling(5).fillna(0).sum().shift()]

        the_stats = pd.concat(a, axis=1)

        the_stats.columns = [
            '%s_%s__back_5_mean__%s' % (prefix, target_column, "#".join(group_column)),
            '%s_%s__back_5_std__%s' % (prefix, target_column, "#".join(group_column)),
            '%s_%s__back_5_sum__%s' % (prefix, target_column, "#".join(group_column)),
            '%s_%s__forward_5_mean__%s' % (prefix, target_column, "#".join(group_column)),
            '%s_%s__forward_5_std__%s' % (prefix, target_column, "#".join(group_column)),
            '%s_%s__forward_5_sum__%s' % (prefix, target_column, "#".join(group_column))
            # '%s_%s__std_by__%s' % (prefix,target_column, "#".join(group_column))
        ]

    return pd.concat([df_old, the_stats])


def get_stats_target(df, group_column, target_column_list, tgt_stats=['mean', 'median', 'max', 'min', 'std'],
                     prefix="hehe", drop_count=True, drop_raw_col=True, filter_count=10):
    df_old = df.copy()
    grouped = df_old.groupby(group_column)
    grouped.order_id.count()
    rs = []
    for target_column in target_column_list:
        def gene_col_names(stat):
            return '{}__{}__{}_by__{}'.format(prefix, target_column, stat, "#".join(group_column))

        the_stats = grouped[target_column].agg(tgt_stats).reset_index()
        # print the_stats.columns
        the_stats.columns = group_column + map(gene_col_names, tgt_stats)
        if drop_raw_col:
            the_stats.drop(group_column, axis=1, inplace=True)
        rs.append(the_stats)

    count = grouped["order_id"].agg(['count']).reset_index()
    count_col_name = '{}__{}__{}_by__{}'.format(prefix, "order_id", "count", "#".join(group_column))
    count.columns = group_column + [count_col_name]
    agg_rs = pd.concat(rs + [count], axis=1)
    af_flt = agg_rs[agg_rs[count_col_name] >= filter_count]

    print "prefix is {},group column is {},agg total count is {}; after filter , agg total count is {}, filter ratio is {} %". \
        format(prefix, "#".join(group_column), len(agg_rs), len(af_flt),
               (1 - len(af_flt) / (len(agg_rs) + 0.01)) * 100)

    if drop_count:
        return af_flt[[x for x in af_flt.columns if "count" not in x]]
    else:
        return af_flt


def ka_add_groupby_features_n_vs_1(df, group_columns_list, target_columns_list, methods_list, keep_only_stats=True,
                                   verbose=1):
    '''Create statistical columns, group by [N columns] and compute stats on [1 column]

       Parameters
       ----------
       df: pandas dataframe
          Features matrix
       group_columns_list: list_like
          List of columns you want to group with, could be multiple columns
       target_columns_list: list_like
          column you want to compute stats, need to be a list with only one element
       methods_list: list_like
          methods that you want to use, all methods that supported by groupby in Pandas

       Return
       ------
       new pandas dataframe with original columns and new added columns

       Example
       -------
       ka_add_stats_features_n_vs_1(train, group_columns_list=['x0'], target_columns_list=['x10'])
    '''
    with tick_tock("add stats features", verbose):
        dicts = {"group_columns_list": group_columns_list, "target_columns_list": target_columns_list,
                 "methods_list": methods_list}

        for k, v in dicts.items():
            try:
                if type(v) == list:
                    pass
                else:
                    raise TypeError(k + "should be a list")
            except TypeError as e:
                print(e)
                raise

        grouped_name = ''.join(group_columns_list)
        target_name = ''.join(target_columns_list)
        combine_name = [[grouped_name] + [method_name] + [target_name] for method_name in methods_list]

        df_new = df.copy()
        grouped = df_new.groupby(group_columns_list)

        the_stats = grouped[target_name].agg(methods_list).reset_index()
        the_stats.columns = [grouped_name] + \
                            ['_%s_%s_by_%s' % (grouped_name, method_name, target_name) \
                             for (grouped_name, method_name, target_name) in combine_name]
        if keep_only_stats:
            return the_stats
        else:
            df_new = pd.merge(left=df_new, right=the_stats, on=group_columns_list, how='left')
        return df_new


class F1Optimizer():
    def __init__(self):
        pass

    @staticmethod
    def get_expectations(P, pNone=None):
        expectations = []
        P = np.sort(P)[::-1]

        n = np.array(P).shape[0]
        DP_C = np.zeros((n + 2, n + 1))
        if pNone is None:
            pNone = (1.0 - P).prod()

        DP_C[0][0] = 1.0
        for j in range(1, n):
            DP_C[0][j] = (1.0 - P[j - 1]) * DP_C[0, j - 1]

        for i in range(1, n + 1):
            DP_C[i, i] = DP_C[i - 1, i - 1] * P[i - 1]
            for j in range(i + 1, n + 1):
                DP_C[i, j] = P[j - 1] * DP_C[i - 1, j - 1] + (1.0 - P[j - 1]) * DP_C[i, j - 1]

        DP_S = np.zeros((2 * n + 1,))
        DP_SNone = np.zeros((2 * n + 1,))
        for i in range(1, 2 * n + 1):
            DP_S[i] = 1. / (1. * i)
            DP_SNone[i] = 1. / (1. * i + 1)

        for k in range(n + 1)[::-1]:
            f1 = 0
            f1None = 0
            for k1 in range(n + 1):
                f1 += 2 * k1 * DP_C[k1][k] * DP_S[k + k1]
                f1None += 2 * k1 * DP_C[k1][k] * DP_SNone[k + k1]
            for i in range(1, 2 * k - 1):
                DP_S[i] = (1 - P[k - 1]) * DP_S[i] + P[k - 1] * DP_S[i + 1]
                DP_SNone[i] = (1 - P[k - 1]) * DP_SNone[i] + P[k - 1] * DP_SNone[i + 1]
            expectations.append([f1None + 2 * pNone / (2 + k), f1])

        return np.array(expectations[::-1]).T

    @staticmethod
    def maximize_expectation(P, pNone=None):
        expectations = F1Optimizer.get_expectations(P, pNone)

        ix_max = np.unravel_index(expectations.argmax(), expectations.shape)
        max_f1 = expectations[ix_max]

        predNone = True if ix_max[0] == 0 else False
        best_k = ix_max[1]

        return best_k, predNone, max_f1

    @staticmethod
    def _F1(tp, fp, fn):
        return 2 * tp / (2 * tp + fp + fn)

    @staticmethod
    def _Fbeta(tp, fp, fn, beta=1.0):
        beta_squared = beta ** 2
        return (1.0 + beta_squared) * tp / ((1.0 + beta_squared) * tp + fp + beta_squared * fn)


import multiprocessing


def applyParallel(dfGrouped, func):
    retLst = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(group) for name, group in dfGrouped)

    return pd.concat(retLst)


def get_rolling_target(df, group_column, target_column_list, prefix="hehe", drop_raw_col=False):
    df_old = df.copy()
    grouped = df_old.groupby(group_column)
    rs = []
    for target_column in target_column_list:
        #         print grouped[target_column].head(5)
        the_stats = [
            grouped[target_column].apply(lambda x: x.rolling(10).mean().shift(-10)),
            grouped[target_column].apply(lambda x: x.rolling(10).std().shift(-10)),
            grouped[target_column].apply(lambda x: x.rolling(10).mean().shift()),
            grouped[target_column].apply(lambda x: x.rolling(10).std().shift()),
        ]

        #         print the_stats.head(20)
        columns = [
            '%s__%s__back_10_mean__%s' % (prefix, target_column, "#".join(group_column)),
            '%s__%s__back_10_std__%s' % (prefix, target_column, "#".join(group_column)),
            '%s__%s__forward_5_mean__%s' % (prefix, target_column, "#".join(group_column)),
            '%s__%s__forward_5_std__%s' % (prefix, target_column, "#".join(group_column)),
        ]
        for n, col in enumerate(columns):
            df_old[col] = the_stats[n]
            #         rs.append(the_stats)

    return df_old


def create_products_faron(df):
    # print(df.product_id.values.shape)
    products = df.product_id.values
    prob = df.reorder_predict.values
    # print "df.reorder_predict.products", products
    sort_index = np.argsort(prob)[::-1]
    L2 = products[sort_index]
    P2 = prob[sort_index]

    opt = F1Optimizer.maximize_expectation(P2)

    best_prediction = ['None'] if opt[1] else []
    best_prediction += list(L2[:opt[0]])

    # print("Prediction {} ({} elements) yields best E[F1] of {}".format(best_prediction, len(best_prediction), opt[2]))
    # print('iteration', df.shape[0], 'optimal value', opt[0],'opt',opt)
    # print

    best = ' '.join(map(lambda x: str(x), best_prediction))
    # print df
    df = df[0:1]
    df.loc[:, 'products'] = best
    return df

dates={}
def lookup(s):
    global dates
    """
    This is an extremely fast approach to datetime parsing.
    For large data, the same dates are often repeated. Rather than
    re-parse these, we store all unique dates, parse them, and
    use a lookup to convert all dates.
    """
    for date in s.unique():

        dates[date] = dates[date] if date in dates else pd.to_datetime(date)
#     dates = {date: pd.to_datetime(date) for date in s.unique()}
    return s.map(dates)

# def create_products_faron(df):
#     products = df.product_id.values
#     prob = df.reorder_predict.values
#
#     sort_index = np.argsort(prob)[::-1]
#     prob = prob[sort_index]
#     products = products[sort_index]
#
#     opt = F1Optimizer.maximize_expectation(prob)
#
#     best_prediction = ['None'] if opt[1] else []
#     best_prediction += [str(p) for p in products[:opt[0]]]
#     f1_max = opt[2]
#
#     best = ' '.join(best_prediction)
#     return (df.iloc[0,0], best)

'''
future method
df2 = df[::-1]
df2.index = pd.datetime(2050,1,1) - df2.index
df2 = df2.rolling('1H').mean()
df3 = df2[::-1]
df3.index = df.index
'''

if __name__ == '__main__':
    fast_search([0.4, 0.3, ])
