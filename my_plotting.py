import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle

class ForecastAnalysis:
    def __init__(self, pkl_path, output_dir):
        self.pkl_path = pkl_path
        self.output_dir = output_dir
        self.csv_path = ""
        self.data = pd.DataFrame()

    def load_pkl(self):
        """Lädt die .pkl-Datei und speichert sie als CSV."""
        try:
            with open(self.pkl_path, "rb") as file:
                data = pickle.load(file)
                self.data = pd.DataFrame(data)
                self.csv_path = os.path.join(self.output_dir, "test_results.csv")
                self.data.to_csv(self.csv_path, index=False)
                print(f"Pickle-Datei erfolgreich geladen und als CSV gespeichert: {self.csv_path}")
        except Exception as e:
            print(f"Fehler beim Laden der .pkl-Datei: {e}")

    def load_csv(self):
        """Lädt die Daten aus der CSV-Datei."""
        try:
            self.data = pd.read_csv(self.csv_path)
            print(f"CSV-Datei erfolgreich geladen: {self.csv_path}")
        except Exception as e:
            print(f"Fehler beim Laden der CSV-Datei: {e}")

    def map_columns_and_filter(self, standardization_params_file):
        # Einlesen der Standardisierungsparameter
        params = pd.read_csv(standardization_params_file)
        features = params['feature'].tolist()

        # Spalten, die gemappt werden sollen
        forecast_columns = [col for col in self.data.columns if col.startswith("Forecast_")]
        true_columns = [col for col in self.data.columns if col.startswith("True_")]

        # Überprüfen, ob die Anzahl der Spalten übereinstimmt
        if len(forecast_columns) != len(features) or len(true_columns) != len(features):
            raise ValueError("Die Anzahl der Forecast- oder True-Spalten stimmt nicht mit den Features überein.")

        # Mapping für Forecast- und True-Spalten erstellen
        forecast_mapping = {col: f"{feature}_Forecast" for col, feature in zip(forecast_columns, features)}
        true_mapping = {col: f"{feature}_True" for col, feature in zip(true_columns, features)}

        # Spalten umbenennen
        self.data.rename(columns={**forecast_mapping, **true_mapping}, inplace=True)

        # Nur gemappte Forecast- und True-Spalten behalten
        relevant_columns = list(forecast_mapping.values()) + list(true_mapping.values())
        self.data = self.data[relevant_columns]

        print("Mapping abgeschlossen und Forecast- sowie True-Spalten behalten:")
        print("Forecast-Mapping:", forecast_mapping)
        print("True-Mapping:", true_mapping)
        print(f"Übrige Spalten nach dem Filtern: {list(self.data.columns)}")

    def destandardize_forecasts(self, standardization_params_file):
        """Destandardisiert Forecast- und True-Werte basierend auf den Standardisierungsparametern."""
        # Einlesen der Standardisierungsparameter aus der CSV-Datei
        params = pd.read_csv(standardization_params_file)

        # Erstellen eines Dictionaries mit den Min- und Max-Werten
        min_values = params.set_index('feature')['min'].to_dict()
        max_values = params.set_index('feature')['max'].to_dict()

        # Spalten auswählen
        forecast_columns = [col for col in self.data.columns if '_Forecast' in col]
        true_columns = [col for col in self.data.columns if '_True' in col]

        # Destandardisierung der Forecast- und True-Spalten
        for column in forecast_columns + true_columns:
            # Extrahieren des Feature-Namens aus dem Spaltennamen
            feature_name = column.replace('_Forecast', '').replace('_True', '')

            # Überprüfen, ob das Feature in den Standardisierungsparametern existiert
            if feature_name in min_values and feature_name in max_values:
                min_val = min_values[feature_name]
                max_val = max_values[feature_name]

                # Destandardisierung der jeweiligen Spalte
                self.data[column] = self.data[column] * (max_val - min_val) + min_val
                print(f"Destandardized {column} with min={min_val} and max={max_val}")

        print("Destandardisierung abgeschlossen:")
        print(self.data.head())



    def save_forecast_true_csv(self):
        """Speichert Forecast und True Values in einer separaten CSV-Datei."""


        self.data.to_csv(os.path.join(output_dir, "forecast_true.csv"), index=False)


    def save_residuals_csv(self):
        """Berechnet und speichert Residuen in einer separaten CSV-Datei."""
        if self.data is None:
            print("Keine Daten geladen. Bitte lade zuerst die Daten.")
            return

        forecast_columns = [col for col in self.data.columns if col.startswith("Forecast_")]
        true_columns = [col for col in self.data.columns if col.startswith("True_")]

        if not forecast_columns or not true_columns:
            print("Forecast- oder True-Werte-Spalten fehlen in den Daten.")
            return

        residuals = self.data[true_columns].values - self.data[forecast_columns].values
        residuals_df = pd.DataFrame(residuals, columns=[f"Residual_{col[9:]}" for col in forecast_columns])
        output_path = os.path.join(self.output_dir, "residuals.csv")
        residuals_df.to_csv(output_path, index=False)
        print(f"Residuen erfolgreich gespeichert: {output_path}")

    def plot_forecast_true(self):
        """Erstellt Plots für Forecast- und True Values."""
        if self.data is None:
            print("Keine Daten geladen. Bitte lade zuerst die Daten.")
            return

        forecast_columns = [col for col in self.data.columns if col.startswith("Forecast_")]
        true_columns = [col for col in self.data.columns if col.startswith("True_")]

        if not forecast_columns or not true_columns:
            print("Forecast- oder True-Werte-Spalten fehlen in den Daten.")
            return

        for f_col, t_col in zip(forecast_columns, true_columns):
            plt.figure()
            plt.plot(self.data[f_col], label="Forecast", linestyle="--")
            plt.plot(self.data[t_col], label="True Value", alpha=0.7)
            plt.title(f"Forecast vs True Value: {f_col[9:]}")
            plt.legend()
            plt.xlabel("Time Step")
            plt.ylabel("Value")
            output_path = os.path.join(self.output_dir, f"plot_forecast_true_{f_col[9:]}.png")
            plt.savefig(output_path)
            plt.close()
            print(f"Plot gespeichert: {output_path}")

    def plot_residuals(self):
        """Erstellt Plots für Residuen."""
        if self.data is None:
            print("Keine Daten geladen. Bitte lade zuerst die Daten.")
            return

        forecast_columns = [col for col in self.data.columns if col.startswith("Forecast_")]
        true_columns = [col for col in self.data.columns if col.startswith("True_")]

        if not forecast_columns or not true_columns:
            print("Forecast- oder True-Werte-Spalten fehlen in den Daten.")
            return

        residuals = self.data[true_columns].values - self.data[forecast_columns].values
        residuals_df = pd.DataFrame(residuals, columns=[f"Residual_{col[9:]}" for col in forecast_columns])

        for col in residuals_df.columns:
            plt.figure()
            plt.plot(residuals_df[col], label="Residuals", color="orange")
            plt.title(f"Residuen: {col[9:]}")
            plt.legend()
            plt.xlabel("Time Step")
            plt.ylabel("Residual Value")
            output_path = os.path.join(self.output_dir, f"plot_residuals_{col[9:]}.png")
            plt.savefig(output_path)
            plt.close()
            print(f"Residual-Plot gespeichert: {output_path}")



if __name__ == "__main__":

    output_dir = os.path.join('plots', 'INDIVIDUAL3/l672_e10_bs32')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pkl_file = os.path.join("output", "INDIVIDUAL3/16012025_125843/test_output.pkl")
    standard_params_file = os.path.join('datasets', "INDIVIDUAL3/processed/INDIVIDUAL3_normalization_params.csv")
    # timestamps = os.path.join('datasets', "INDIVIDUAL1/processed/INDIVIDUAL1_timestamps_test.pkl")
    analyser = ForecastAnalysis(pkl_file, output_dir)
    analyser.load_pkl()
    analyser.load_csv()
    analyser.map_columns_and_filter(standard_params_file)
    analyser.destandardize_forecasts(standard_params_file)
    # analyser.add_timestamps(timestamps)
    analyser.save_forecast_true_csv()
    # analyser.save_residuals_csv()
    # analyser.plot_forecast_true()
    # analyser.plot_residuals()