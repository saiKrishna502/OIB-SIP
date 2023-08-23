import pandas as pd
import tkinter as tk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load your dataset (replace 'sales_data.csv' with your file)
data = pd.read_csv(r"C:\Users\saikr\Downloads\archive (7)\Advertising.csv")

# Split data into features (X) and target (y)
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)


# Create a simple GUI using Tkinter
class SalesPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sales Prediction App")

        self.label = tk.Label(root, text="Enter advertising spend (TV, Radio, Newspaper):")
        self.label.pack()

        self.entry = tk.Entry(root)
        self.entry.pack()

        self.predict_button = tk.Button(root, text="Predict Sales", command=self.predict)
        self.predict_button.pack()

        self.result_label = tk.Label(root, text="")
        self.result_label.pack()

    def predict(self):
        try:
            advertising_spend = self.entry.get().split(',')
            advertising_spend = [float(spend.strip()) for spend in advertising_spend]

            if len(advertising_spend) != 3:
                raise ValueError()

            # Create a DataFrame with the appropriate column names for prediction
            prediction_df = pd.DataFrame([advertising_spend], columns=['TV', 'Radio', 'Newspaper'])

            prediction = model.predict(prediction_df)
            self.result_label.config(text=f"Predicted Sales: {prediction[0]:.2f}")
        except ValueError:
            self.result_label.config(text="Invalid input. Enter three numbers separated by commas.")


# Create the main GUI window
root = tk.Tk()
app = SalesPredictionApp(root)
root.mainloop()
