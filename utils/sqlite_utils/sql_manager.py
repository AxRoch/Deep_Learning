import sqlite3


class SQLManager():
    """Instantiates an instance which allows to safely and quickly load and explore an SQL database using sqlite.
    In particular, this instance safely closes each cursor opened when the asked task is over.

    Args
    ----
    path_db : string, optional
        If given, directly instantiate a connection with the database associated with the corresponding path. 
    """
    
    def __init__(self, path_db=None):
        self.connection = None
        if path_db is not None:
            self.load_db(path_db)
        
        
    def _safe_cursor_op(self, *ops):
        cursor = self.connection.cursor()
        try:
            for op in ops:
                result = op(cursor)
        except Exception as e:
            raise
        finally:
            cursor.close()
        return result
    
    
    def apply(self, table, column, func):
        """Modifies the given table by applying a given function to the given column.

        Args
        ----
        table : string
            The name of the given table to modify.
        column : string
            The name of the column to modify.
        func : callable
            The given function. This function wil be applied to each elements of the given column.
            It takes as argument the value to change and its given index.
        """
        old_values = self.execute_and_fetch(f"SELECT {column}, rowId FROM {table}")
        new_values = [(func(elem, rowId), rowId) for elem, rowId in old_values]
        self.executemany(f'UPDATE {table} SET {column}= ? WHERE rowId= ?', new_values)    
    
    
    def commit(self):
        """Commits all the modification made to the connected SQL database in the associated path.
        """
        self.connection.commit()
        
    
    def close(self):
        """Closes the current connection.
        """
        self.connection.close()
        self.connection = None
        
    
    def execute(self, sql, parameters=()):
        """Execute the given SQL statement.

        Args
        ----
        sql : str
            The given SQL statement to execute
        parameters : dict or sequence, optional
            Values to bind to placeholders in the given statement.
        """
        self._safe_cursor_op(lambda cursor: cursor.execute(sql, parameters))
    
    
    def executemany(self, sql, parameters):
        """Execute the given SQL statement a number of time corresponding to the size of
        `parameters` argument.

        Args
        ----
        sql : str
            The given SQL statement to execute
        parameters : iterable
            Values to bind to placeholders in the given statement.
        """
        self._safe_cursor_op(lambda cursor: cursor.executemany(sql, parameters))
            
        
    def execute_and_fetch(self, sql, parameters=()):
        """Executes the given SQL statement and returns the result.

        Args
        ----
        sql : str
            The given SQL statement to execute
        parameters : dict or sequence, optional
            Values to bind to placeholders in the given statement.
        """
        result = self._safe_cursor_op(
            lambda cursor: cursor.execute(sql, parameters),
            lambda cursor: cursor.fetchall()
        )
        
        return result
    
    
    def get_column_names(self, table_name):
        """Returns the names of all the column of the given table.

        Args
        ----
        table_name : str
            The name of the given table.
        """
        result = self._safe_cursor_op(
            lambda cursor: cursor.execute(f'SELECT * from {table_name}'),
            lambda cursor: cursor.description
        )
        
        return [res[0] for res in result]
    

    def load_db(self, path_db):
        """Instantiates a connection with the database associated with the corresponding path.

        Args
        ----
        path_db : string, optional
            The given path.
        """
        if self.connection is not None:
            if 'y' != input(f'{self.path_db} database already loaded. Closed this database to load the new one ?(y/n)'):
                print('Aborted')
                return
            self.close()
        print(f"Loading {path_db}")
        self.connection = sqlite3.connect(path_db)
        self.path_db = path_db
    
    
    def summary(self):
        """Displays a table representing the different tables of the connected database and their associated column
        names.
        """
        print(f"Summary of {self.path_db}:")
        for table in self.tables:
            table_name = table[0]
            print(table_name, ':')
            print('   ', ' | '.join(self.get_column_names(table_name)))
        
    
    @property
    def tables(self):
        """The list of all the table names.
        """
        return self.execute_and_fetch("SELECT name FROM sqlite_master WHERE type='table'")