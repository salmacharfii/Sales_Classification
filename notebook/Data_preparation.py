from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

def split_data(df, target="sales", val_size=0.15, test_size=0.15, random_state=42):
	"""Split a DataFrame into train, validation and test sets."""
	X = df.drop(columns=target)
	y = df[target]

	temp_size = val_size + test_size
	X_train, X_temp, y_train, y_temp = train_test_split(
		X, y, test_size=temp_size, random_state=random_state
	)

	X_val, X_test, y_val, y_test = train_test_split(
		X_temp, y_temp, test_size=test_size / temp_size, random_state=random_state
	)

	print(X.shape, X_train.shape, X_val.shape, X_test.shape, y.shape)
	return X_train, X_val, X_test, y_train, y_val, y_test


COLS_TO_DROP = ["Unnamed: 0", "date", "store_ID"]

holiday_map = {'0': 0, 'a': 1, 'b': 2, 'c': 3}
categories  = [0, 1, 2, 3]


def encode_state_holiday(df):
    df = df.copy()
    # 1) letters -> numbers
    df['state_holiday'] = (
        df['state_holiday'].astype(str).str.strip().str.lower()
          .replace({'': '0', 'nan': '0'})
          .map(holiday_map)
          .fillna(0)
          .astype(int)
    )
    # 2) one-hot encoding
    df['state_holiday'] = pd.Categorical(df['state_holiday'], categories=categories)
    df = pd.get_dummies(df, columns=['state_holiday'], prefix='state_hol', dtype=int)
    return df


def clean_data(df, cols_to_drop=COLS_TO_DROP):
    """Drop unused columns and encode state_holiday."""
    df = df.drop(columns=cols_to_drop, errors="ignore")
    df = encode_state_holiday(df)
    return df


def clean_splits(X_train, X_val, X_test, cols_to_drop=COLS_TO_DROP):
    """Apply the same cleaning to train, validation and test sets."""
    X_train_clean = clean_data(X_train, cols_to_drop)
    X_val_clean   = clean_data(X_val, cols_to_drop)
    X_test_clean  = clean_data(X_test, cols_to_drop)

    for name, d in [("X_train_clean", X_train_clean),
                    ("X_val_clean", X_val_clean),
                    ("X_test_clean", X_test_clean)]:
        print(f"Unnamed: 0 in {name}:", "Unnamed: 0" in d.columns)

    print(X_train_clean.filter(like='state_hol').sum())
    return X_train_clean, X_val_clean, X_test_clean


def scale_splits(X_train, X_val, X_test):
    """Fit a StandardScaler on train only, then apply it to val and test."""
    scaler = StandardScaler()

    X_train_sc = scaler.fit_transform(X_train)   # learn + apply
    X_val_sc   = scaler.transform(X_val)         # apply only
    X_test_sc  = scaler.transform(X_test)

    print(pd.DataFrame(X_train_sc, columns=X_train.columns).head())
    return X_train_sc, X_val_sc, X_test_sc, scaler