import argparse
import logging

from data_processors import TrainCSVProcessor, IdealCSVProcessor, TestCSVProcessor

# create logger
logging.basicConfig(filename='logs.txt',
                    level=logging.INFO,
                    filemode='w',  # overwrite logger file for every run
                    format="%(asctime)s :: %(levelname)s: %(message)s")
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())


def main(train_csv_path: str, ideals_csv_path: str, test_csv_path: str):
    # init processor classes which does csv reading during initialization
    logger.info("Loading data...")
    train_processor = TrainCSVProcessor(csv_path=train_csv_path)
    ideals_processor = IdealCSVProcessor(csv_path=ideals_csv_path)
    test_processor = TestCSVProcessor(csv_path=test_csv_path)
    logger.info("Finished loading data.")

    # visualize raw datasets
    logger.info("Saving figures of raw data...")
    train_processor.plot_raw_data(plot_title='Train csv', save_path='images/train_csv.jpg')
    ideals_processor.plot_raw_data(plot_title='Ideals csv (subset of columns)',
                                   save_path='images/ideals_csv.jpg', cols_to_plot=['y10', 'y14', 'y15', 'y42'])
    test_processor.plot_raw_data(plot_title='Test csv', save_path='images/test_csv.jpg', plot_type='scatter')

    # overlay test csv on train plot
    train_processor.plot_multiple_dataframes(list_of_dfs=[train_processor.data,
                                                          test_processor.data.sort_values(by=['x'])],
                                             cols_to_plot=[train_processor.y_cols, test_processor.y_cols],
                                             plot_types=['plot', 'scatter'],
                                             plot_title='Test csv over train csv',
                                             save_path='images/train_and_test.jpg')
    logger.info("Finished saving figures.")

    # match columns of ideal to columns in train and visualize the result
    logger.info("Matching ideals with train csv columns...")
    train_processor.match_to_ideals(ideals_df=ideals_processor.data)
    train_processor.vis_mapping(ideals_df=ideals_processor.data,
                                plot_title="Closest Ideal functions to train",
                                save_path='images/matched_ideals_to_train.jpg')
    logger.info("Finished matching with train.")

    # Now lets match x,y pairs from the test dataset to estimated ideal functions
    logger.info("Matching ideals with test x,y pairs...")
    test_processor.assign_to_ideals(train_processor=train_processor, ideals_processor=ideals_processor)
    # Plot test set and estimated closest ideals
    test_processor.plot_with_assigned(plot_title="Graph of values from test.csv and closes ideals", sort_by='x',
                                      save_path='images/test_assigned.jpg')
    logger.info("Finished matching with test")

    # Now save the tables into a sqlite db
    logger.info("Creating database tables...")
    train_processor.save_to_sql(file_path="database/training", suffix=" (training func)")
    ideals_processor.save_to_sql(file_path="database/ideal", suffix=" (ideal func)")
    test_processor.save_to_sql(file_path="database/assigned", suffix="",
                               rename_columns={'x': 'X (test func)', 'test': 'Y (test func)'},
                               chunksize=1)  # chunksize=1 writes line by line
    logger.info("Finished creating database tables.")
    return 0


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv_path', type=str, default='datasets/train.csv',
                        help='Path to the training csv file')
    parser.add_argument('--ideals_csv_path', type=str, default='datasets/ideal.csv',
                        help='Path to the training csv file')
    parser.add_argument('--test_csv_path', type=str, default='datasets/test.csv',
                        help='Path to the training csv file')

    return parser.parse_args()


if __name__ == '__main__':
    args = argument_parser()
    try:
        main(train_csv_path=args.train_csv_path, ideals_csv_path=args.ideals_csv_path, test_csv_path=args.test_csv_path)
    except Exception as e:
        logger.exception("main crashed. Error: %s", e)
    else:
        logger.info("Program finished successfully.")

