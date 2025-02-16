import polars as pl

def main():
    df = pl.read_csv('data/collated_results.csv')
    print(df.select('solve_rate').group_by('solve_rate').count().sort('solve_rate'))

    for row in df.filter(pl.col('solve_rate') == 0.0).to_dicts():
        print(row['question'], '\n')


if __name__ == '__main__':
    main()