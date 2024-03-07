# Python course project

### How to run

To run the code first install requirements.txt via command:
```bash
pip install -r requirements.txt
```
After this you can run python code via:
```python
python main.py
```

This will generate dataset plots, sqlite database files and logs.txt file in the root folder.
_But maybe you need to delete them before running as they already exist in the repository_.

To run tests located under `test` module you need to run:
```python
python -m unittest discover -v
```

### Code structure

The main logic which runs all the necessary classes/functions is located in `main.py` file.
After executing `main.py` file, you will see in the console the execution logs about underlying processes, also
generated `images` and `database` folders and `logs.txt` file which contains the same logs which you will see in the 
console.

The `images` folder contains generated figures about of the datasets, and `database` folder contains generated sqlite 
files with the tables requested by assignment description.

The `data_processors` module contains the main classes which process the csv files to do matching between train and 
ideals csv files and also assign each x,y pairs from the test csv to one of the functions from ideals. All this processing
logic is split between `TrainCSVProcessor`, `IdealCSVProcessor` and `TestCSVProcessor` data processor classes
and each class contains corresponding functions to execute the logic. The function call of the corresponding functions
is located in `main.py` as described above.

***`homework.ipynb`*** is a jupyter notebook file which contains the original code which I wrote before moving to class
based code structure. For me it was easier to do original data exploratory analysis and first version of the code in the 
jupyter notebook as you can see the resulting dataframes of your code immediately under code cell in jupyter notebook.

`test` modules contains unit test class for testing the code and `datasets` dir contains the csv files.

***Please do not delete files in `datasets` folder***, as the test cases are using those csv files for code 
logic testing. 


