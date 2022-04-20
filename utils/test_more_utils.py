from unittest.mock import Mock

import pandas as pd
import pytest

from utils.more_utils import find_column, find_columns, has_non_trivial_return_value


@pytest.fixture
def df():
    return pd.DataFrame({'a': [1, 2], 'b': [10, 20]})


def test_find_columns(df):
    assert find_columns(df, regex='a') == ['a']


def test_columns_uses_filter():
    df = Mock()
    kwargs_for_filter = {'foo': 'bar'}

    find_columns(df, **kwargs_for_filter)

    df.filter.assert_called_with(**kwargs_for_filter)


def test_find_column(df):
    assert find_column(df, regex='a') == 'a'


@pytest.mark.parametrize('regex', ['c', 'a|b'])
def test_find_column_raises_value_error_if_not_exactly_one_found(df, regex):
    with pytest.raises(ValueError):
        find_column(df, regex=regex)


@pytest.mark.parametrize('fun',
[
    lambda: pd.Series(index=[0, 1], name='foo', dtype='object'),
    lambda: pd.DataFrame(),
    lambda: pd.DataFrame(index=[0, 1], columns=['foo', 'bar']),
], ids=[
    'All null Series',
    'Empty DataFrame',
    'All null DataFrame',
    ])
def test_has_non_trivial_return_value_raises_value_error(fun):
    decorated = has_non_trivial_return_value(fun)
    with pytest.raises(ValueError, match='Trivial'):
        decorated()
