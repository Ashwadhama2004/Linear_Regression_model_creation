# Linear_Regression_model_creation



# 💼 Linear Regression Salary Prediction System 💸

Welcome to the **Linear Regression Salary Prediction System**! 🎉 This project helps predict employee salaries based on their work experience 📊 using linear regression with gradient descent. Whether you're a hiring manager or just curious about salary trends, we’ve got you covered! 🕵️‍♂️💰

---

## 🔍 Project Overview

This project implements a simple yet powerful **Linear Regression** model to predict the salary of an employee based on their years of experience. The model learns the relationship between experience and salary by fitting a line through the data and then predicts the salary of new employees using that line! 📈👩‍💻

### 📊 Dataset Features

Our dataset includes two main columns:

- **Work Experience**: The number of years an employee has worked. 🏢
- **Salary**: The employee's salary in dollars 💵 (our target variable to predict).

---

## 🛠️ How It Works

Using the **Gradient Descent** algorithm, we minimize the loss between the predicted salaries and the actual salaries by updating the **weight** (slope) and **bias** (intercept) of our linear equation until it fits the data perfectly. The goal is to predict future salaries based on the weight of the work experience.

### Key Equation:
- **Prediction Formula**: `Y = wX + b`
  - **Y**: Salary (Dependent Variable)
  - **X**: Work Experience (Independent Variable)
  - **w**: Weight (Slope of the line)
  - **b**: Bias (Intercept)

---

## 🚀 How to Run the Project

Wanna predict some salaries? Follow these easy steps:

### Prerequisites

First, make sure you’ve got the necessary Python libraries installed. You can do this by running:

```bash
pip install numpy pandas matplotlib scikit-learn
```

### Step-by-Step Guide

1. **Clone the Repo**: Download this project to your computer:

```bash
git clone https://github.com/Ashwadhama2004/linear-regression-salary-prediction.git
cd linear-regression-salary-prediction
```

2. **Get the Dataset**: Ensure that the `salary_data.csv` file is in your project folder. If you don’t have it, download it from [here](#).

3. **Run the Code**: You can run the code using a Jupyter notebook or a Python script:

```bash
jupyter notebook salary_prediction.ipynb
```

Or if you're in a hurry, just run the Python script:

```bash
python salary_prediction.py
```

---

## 🏗️ Code Breakdown

Here’s what’s happening inside the code:

1. **Data Loading**:
   - We load the **salary_data.csv** dataset to get the experience and salary information.

2. **Splitting the Data**:
   - We split the data into training and testing sets (67% for training, 33% for testing).

3. **Model Building**:
   - The `Linear_Regression` class is created using the equation `Y = wX + b`.
   - The **gradient descent** algorithm is implemented to update weights and bias.

4. **Visualization**:
   - Using **matplotlib**, we plot both the training and testing data, visualizing the predicted salary vs actual salary to see how well the model performs.

5. **Performance Evaluation**:
   - We calculate the **R-squared** score, a measure of how well our model explains the variance in the data.

---

## 🖥️ Sample Code

Wanna peek at some sample code? Here you go:

```python
# Initialize the model
model = Linear_Regression(learning_rate=0.01, no_of_iterations=1000)

# Fit the model with training data
model.fit(X_train, Y_train)

# Predict the salary for the test data
test_data_prediction = model.predict(X_test)

# Plot the results
plt.scatter(X_test, Y_test, color='red')
plt.plot(X_test, test_data_prediction, color='blue')
plt.xlabel('Work Experience')
plt.ylabel('Salary')
plt.title('Salary vs Experience')
plt.show()
```

---

## 📈 Results

The model makes accurate predictions of salaries based on work experience. It’s a simple, efficient model that provides insights into salary trends! 🎉

Here’s an example of how the model fits the data and the performance metrics:

- **R-squared Score for Training Data**: 0.95 💯
- **R-squared Score for Testing Data**: 0.92 🚀

The **closer to 1**, the better! Our model is doing pretty well!

---

## 🌟 Future Enhancements

Want to make the model even cooler? Here are some ideas:

- **Add More Features**: Incorporate additional features like **education level**, **industry**, or **location** for a more robust prediction.
- **Implement Regularization**: Add **L1/L2 regularization** to reduce overfitting and improve generalization.
- **Use Polynomial Regression**: Fit a polynomial curve to capture more complex relationships between experience and salary.

---

## 🛠️ Tech Stack

- **Language**: Python 🐍
- **Libraries**: `numpy`, `pandas`, `matplotlib`, `scikit-learn`

---

## 📜 License

This project is licensed under the MIT License. For more details, check the [LICENSE](LICENSE) file.

---

## 👋 Connect with Me

Got questions or feedback? I’d love to hear from you!

- GitHub: [Ashwadhama2004](https://github.com/Ashwadhama2004)
- LinkedIn:(www.linkedin.com/in/gaurang-chaturvedi-390ab5221)

---

Thanks for checking out my **Linear Regression Salary Prediction System**! Happy coding! 😊
