# Data Cleaning, Processing & Visualization Techniques — Summary

This document is a **consolidated reference** of all data cleaning, processing, and visualization techniques used in three notebooks: **exportdata.ipynb** (preprocessing for ML), **ventas.ipynb** (sales EDA), and **example.ipynb** (California Housing, Hands-On ML). Each technique is briefly defined and linked to its source notebook(s).

---

## 1. Data cleaning

| Technique | Summary | Source(s) |
|-----------|---------|-----------|
| **Drop empty rows** | Remove rows where all cells are NaN: `dropna(how='all')`. | ventas |
| **Filter by string condition** | Remove bad rows (e.g. repeated headers) with a logical mask, e.g. `data_anual['Order Date'].str[0:2] != 'Or'`. | ventas |
| **Imputation (mean)** | Fill missing numeric values with the column mean using `SimpleImputer(strategy='mean')`; fit on non-missing data, then transform. | exportdata |

---

## 2. Data processing

### 2.1 Data consolidation and loading

- **Multi-file concatenation**: Loop over CSV files, read with `pd.read_csv`, merge with `pd.concat` into one DataFrame; optionally export to a single CSV (ventas).
- **Load and slice**: `pd.read_csv`, then `iloc` for features (e.g. `iloc[:, :-1]`) and target (e.g. `iloc[:, 3]`) (exportdata).
- **Sklearn dataset**: `fetch_california_housing(as_frame=True)` to get a Bunch with `.frame` as DataFrame (example).

### 2.2 Feature engineering

- **Derived numeric column**: New column from existing ones, e.g. `Sales = Quantity Ordered * Price Each`; use `astype('int')` / `astype('float')` if needed (ventas).
- **Extract from string**: Month from date string via `Order Date.str[:2]` and `pd.to_numeric`; city from address via custom function and `.apply()` (ventas).
- **Binning (discretization)**: Convert continuous variable to categories with `pd.cut(bins=..., labels=...)` for stratification or grouping (example).

### 2.3 Encoding

- **Label encoding**: Map categories to integers with `LabelEncoder`; good for target (e.g. Yes/No → 0/1); for features can imply false order (exportdata).
- **One-hot encoding**: Replace one categorical column with dummy (0/1) columns via `OneHotEncoder(sparse_output=False)`; use with `np.column_stack` to keep numeric columns (exportdata).

### 2.4 Train/test splitting

- **Random split (manual)**: Shuffle indices with `np.random.permutation`, take first k% as test, rest as train (example).
- **Random split (sklearn)**: `train_test_split(df, test_size=0.2, random_state=42)` for reproducible holdout (example, exportdata).
- **Stratified split**: Preserve proportion of a categorical variable (e.g. income category) in train and test with `StratifiedShuffleSplit` (example).
- **Deterministic split by ID**: Hash row ID (e.g. CRC32) to assign train/test so same ID always in same set when data is updated (example).

### 2.5 Post-split hygiene

- **Drop temporary columns**: Remove columns used only for splitting (e.g. `income_cat`) from both train and test (example).
- **Work on train only**: Set `df = strat_train_set.copy()` so EDA and feature work use only training data and avoid leakage (example).

### 2.6 Scaling

- **Standardization (z-score)**: `StandardScaler` — fit on train, transform train and test; features get mean 0, std 1 (exportdata).
- **Concept of normalization**: Min-max scaling to [0,1] (formula mentioned in exportdata; StandardScaler used in code).

---

## 3. Visualization

### 3.1 Univariate / distributions

- **Histograms**: `df.hist(bins=50)` for each numeric column to see distribution, skew, scale (example).
- **Histogram of categorical/binned**: e.g. `df["income_cat"].hist()` to check category distribution (example).

### 3.2 Bivariate / spatial

- **Geographic scatter**: Longitude vs latitude with `df.plot(kind="scatter", x="Longitude", y="Latitude", alpha=0.1)` for density (example).
- **Geographic scatter with size and color**: Same axes with point size proportional to one variable (e.g. population) and color to another (e.g. MedHouseVal) via `c=`, `s=`, `cmap` (example).

### 3.3 Correlation and multivariate EDA

- **Correlation matrix**: `df.corr()` for pairwise Pearson correlation; sort by target column to see feature–target strength (example).
- **Scatter matrix (pair plot)**: `scatter_matrix(df[attributes])` for selected columns; off-diagonal scatter, diagonal distribution (example).

### 3.4 Aggregation-based plots

- **Bar chart (by category)**: `groupby` + `sum()` or `count()`, then `plt.bar(categories, values)` — e.g. sales by month, sales by city (ventas). Use `.index` and `.values` from the groupby result for correct bar positions.
- **Line chart (time/sequence)**: `groupby` (e.g. by hour) + `count()` or `sum()`, then `plt.plot(positions, values)` for orders by hour (ventas).

### 3.5 Other processing for visualization

- **Grouping and aggregation**: `groupby('Column').sum()`, `groupby('Column').count()` to prepare counts or totals for bar/line charts (ventas).
- **Products bought together**: Create concatenated product list per order with `groupby('Order ID')['Product'].transform(lambda x: ', '.join(x))`, then `drop_duplicates`, then count pairs with `itertools.combinations` and `Counter` (ventas).
