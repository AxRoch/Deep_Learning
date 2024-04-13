# SQL Utils

This package introduces an `SQLManager` class which creates instances which allowing to safely and quickly load and explore an SQL database using the `sqlite3` package. In particular, this instance safely closes each cursor opened when the asked task is over, even if an error occurs.
This can be especially useful when working with notebooks when some opened cursors might be overwritten before being closed.
Also, this enables to call SQL command with a more user-friendly interface.

## Getting started

The first step is to instantiate an `SQLManager` object and to connect it to an SQL database.
This can be done either in two lines:
```python
sql = SQLManager()
sql.load_db('path/to/my/database.db')
```
or with one line:
```python
sql = SQLManager('path/to/my/database.db')
```

The `close` method is used to end the connection with the database:
```python
sql.close()
```

## Some useful features

The class offers convenient methods.

First, the `tables` property gives the list of all tables in the database.

The `get_column_names` returns the list of the columns of the given table.

Finally, the `summary` method gives an overview of all the tables of the database and their associated columns.


## Example

Here are 2 equivalent examples without and with `SQLManager` class to create a database with a table `table1(col1, col2, col3)`, then to fill it and finally to modify the elements of the last column.

### Baseline : without SQLManager class
```python
import sqlite3

# Load database
path_db = 'path/to/my/database.db'
connection = sqlite3.connect(path_db)
cursor = connection.cursor()

# Create a table
cursor.execute('CREATE TABLE table1(col1, col2, col3)')

# Insert rows in the table
cursor.execute('INSERT INTO table1 VALUES(1, 2, 3)')
cursor.execute('INSERT INTO table1 VALUES(10, 20, 30)')
cursor.execute('INSERT INTO table1 VALUES(1, 20, 300)')

# Retrieve the elements of the first column
col1_cursor = connection.cursor()
col1_cursor.execute('SELECT col1 FROM table1')
col1 = col1_cursor.fetchall()
col1_cursor.close()

# Add the value of the elements of the first column to the third column
col3_cursor = connection.cursor()
col3_cursor.execute('SELECT col3, rowId FROM table1')
col3_old = col3_cursor.fetchall()
col3_cursor.close()

col3_new = [(elem + col1[rowId-1][0], rowId) for elem, rowId in col3_old]
cursor.executemany(f'UPDATE table1 SET col3= ? WHERE rowId= ?', col3_new)

# Commit the modifications
cursor.close()
connection.commit()
connection.close()
```

### Example with SQLManager class
The following code gives an example of how to use `SQLManager` class to get equivalent result:
```python
import sql_utils

# Load database
path_db = 'path/to/my/database.db'
sql = sql_utils.SQLManager(path_db)

# Create a table
sql.execute('CREATE TABLE table1(col1, col2, col3)')

# Insert rows in the table
sql.execute('INSERT INTO table1 VALUES(1, 2, 3)')
sql.execute('INSERT INTO table1 VALUES(10, 20, 30)')
sql.execute('INSERT INTO table1 VALUES(1, 20, 300)')

# Retrieve the elements of the first column
col1 = sql.execute_and_fetch('SELECT col1 FROM table1')

# Add the value of the elements of the first column to the third column
sql.apply(table='table1',
          column='col3',
          func=lambda value, idx: col1[idx-1][0] + value)

# Commit the modifications
sql.commit()
```

If the amount of data to fetch is too large to be completely loaded, the `iter` method can be used
instead of `execute_and_fetch`. This method creates an iterator that iterates over the results of
the specified sql statement. For example:

```python
total = 0
for elem in sql.iter('SELECT col1 FROM table1'):
    total += elem
```

For other specific applications, the class also provides a `safe_cursor` method which returns a
context manager enabling to create a cursor that will be automatically closed once the `with`
statement is over:

```python
with sql.safe_cursor() as col1_cursor:
    col1_cursor.execute('SELECT col1 FROM table1')
    col1 = col1_cursor.fetchall()
```