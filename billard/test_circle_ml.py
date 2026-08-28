import io
import pandas as pd
import pytest

from billard.circle_ml import read_data

def test_read_data_valid_csv():
    # Mock a CSV file content
    csv_content = "col1,col2\n1,2\n3,4"
    mock_file = io.StringIO(csv_content)

    # Call the function
    df = read_data(mock_file)

    # Verify the result is a DataFrame
    assert isinstance(df, pd.DataFrame)

    # Verify shape and content
    assert df.shape == (2, 2)
    assert list(df.columns) == ['col1', 'col2']
    assert df['col1'].tolist() == [1, 3]
    assert df['col2'].tolist() == [2, 4]

def test_read_data_file_not_found():
    # Verify that passing a non-existent file path raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        read_data("non_existent_file.csv")

def test_read_data_empty_file():
    # Mock an empty CSV file content
    empty_content = ""
    mock_file = io.StringIO(empty_content)

    # Verify that reading an empty file raises pd.errors.EmptyDataError
    with pytest.raises(pd.errors.EmptyDataError):
        read_data(mock_file)
