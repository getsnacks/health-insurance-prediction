# Insurance Charges Prediction
This project tackles a real-world regression problem by predicting insurance charges based on demographic and health data using machine learning techniques. 
## Steps
### Data Cleaning

- Missing values removed.
- Categorical variables like region and sex standardized.
- Smoker column converted to boolean.
- Charges standardized to numeric.
- Children converted to integer.
- All numeric values made positive.

### Model Training

One-hot encoding applied to the region column.
smoker and sex transformed to numerical values.

Linear regression model trained and evaluated using R² score.
### Prediction

Predictions made on the validation dataset, with low predictions corrected.
## Libraries used
- pandas
- numpy
- matplotlib
- sklearn
## How to use
1. **Load and clean the data**  
   ```python
   insurance = pd.read_csv('insurance.csv')

2. **Prepare the Data for Model Fitting**  
   Preprocess the data by converting categorical variables into dummy variables, encoding the smoker column, and converting the sex column into a binary format. Drop any missing values:
   ```python
   df_new = pd.get_dummies(df, prefix=['region'], columns=['region'])
   df_new = df_new.drop(columns=['region_southeast'])
   df_new['smoker'] = df_new['smoker'].astype('int64')
   df_new['is_male'] = (df_new['sex'] == 'male').astype('int64')
   df_new = df_new.drop(columns=['sex'])
   df_new = df_new.dropna()

3. **Fit a Linear Regression Model**  
   Split the data into training and testing sets (80% training and 20% testing). Initialize a linear regression model and fit it on the training data. After training, make predictions on the test set and evaluate the model's performance using the R^2 score:
   ```python
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression

    X = df_new.drop(columns=['charges'])
    y = df_new['charges']

    X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    print(f"R^2 Score: {r2_score}")

4. **Make Predictions**  
   To make predictions on new data:
   ```python
   predictions = model.predict(input_df)

5. **Post-Processing Predictions**  
   To ensure predicted charges are above a threshold:

   ```python
   validation_data['predicted_charges'] = predictions
   validation_data.loc[validation_data['predicted_charges'] < 1000, 'predicted_charges'] = 1000
## License

[MIT](https://choosealicense.com/licenses/mit/)

