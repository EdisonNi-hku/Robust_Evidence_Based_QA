import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, default="")
    args = parser.parse_args()

    df_all = pd.read_csv(args.filename + '_with_citation.csv')
    df_11b_nli = pd.read_csv(args.filename + '_11b_entail.csv')[['id', 'attribution', 'accuracy']]
    df_11b_nli = df_11b_nli.rename(columns={'attribution': 'attribution_11b', 'accuracy': 'accuracy_11b'})
    df_3b_nli = pd.read_csv(args.filename + '_3b_entail.csv')[['id', 'attribution', 'accuracy']]
    df_3b_nli = df_3b_nli.rename(columns={'attribution': 'attribution_3b', 'accuracy': 'accuracy_3b'})
    print(df_all.info())
    df_all = df_all.merge(df_11b_nli, on='id', how='left')
    df_all = df_all.merge(df_3b_nli, on='id', how='left')
    print(df_all.info())

    df_all['accuracy_11b'] = df_all['accuracy_11b'].fillna(0)
    df_all['accuracy_3b'] = df_all['accuracy_3b'].fillna(0)
    df_all['all_faithful'] = df_all.accuracy_3b * df_all.accuracy_11b
    df_all.loc[(df_all.all_faithful == 0) & (df_all.format_error_type == 0), 'format_error_type'] = 4

    df_all.to_csv(args.filename + '_all.csv', index=False)

    faithfulness_score_11b = df_all.groupby('id')['accuracy_11b'].mean()
    faithfulness_score_3b = df_all.groupby('id')['accuracy_3b'].mean()
    faithfulness_score = df_all.groupby('id')['all_faithful'].mean()
    df_result = pd.DataFrame({'11b': faithfulness_score_11b, '3b': faithfulness_score_3b, 'all': faithfulness_score})

    print('11b score: %.2f' % float(100 * df_result['11b'].mean()),
          '3b score: %.2f' % float(100 * df_result['3b'].mean()),
          'agg score: %.2f' % float(100 * df_result['all'].mean()),
          'type 4: %.2f' % (100 * len(df_all.loc[df_all.format_error_type == 4, :]) / len(df_all)),
          'type 3: %.2f' % (100 * len(df_all.loc[df_all.format_error_type == 3, :]) / len(df_all)),
          'type 2: %.2f' % (100 * len(df_all.loc[df_all.format_error_type == 2, :]) / len(df_all)),
          'type 1: %.2f' % (100 * len(df_all.loc[df_all.format_error_type == 1, :]) / len(df_all)),
          )


if __name__ == '__main__':
    main()
