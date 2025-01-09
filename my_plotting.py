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

    def save_forecast_true_csv(self):
        """Speichert Forecast und True Values in einer separaten CSV-Datei."""
        if self.data is None:
            print("Keine Daten geladen. Bitte lade zuerst die Daten.")
            return

        forecast_columns = [col for col in self.data.columns if col.startswith("Forecast_")]
        true_columns = [col for col in self.data.columns if col.startswith("True_")]

        if not forecast_columns or not true_columns:
            print("Forecast- oder True-Werte-Spalten fehlen in den Daten.")
            return

        forecast_true_data = self.data[forecast_columns + true_columns]
        output_path = os.path.join(self.output_dir, "forecast_true_values.csv")
        forecast_true_data.to_csv(output_path, index=False)
        print(f"Forecast und True Values erfolgreich gespeichert: {output_path}")

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
    file_path = "/plots/l96_e10_bs32/data.csv"
    output_dir = "C:/Users/Vika/Documents/HTWG/Local_Thesis/mtad-gat-pytorch/plots"
    pkl_file = "C:/Users/Vika/Documents/HTWG/Local_Thesis/mtad-gat-pytorch/output/MYDATA/08012025_152453/test_output.pkl"
    # Datei öffnen und als DataFrame laden
    # with open(pkl_file, "rb") as file:
    #     df = pickle.load(file)
    #
    # # Überprüfen, was geladen wurde
    # print(df.head())  # Zeige die ersten 5 Zeilen an
    # print(df.info())  # Informationen zum DataFrame
    analyser = ForecastAnalysis(pkl_file, output_dir)
    analyser.load_pkl()
    analyser.load_csv()
    analyser.save_forecast_true_csv()
    analyser.save_residuals_csv()
    analyser.plot_forecast_true()
    analyser.plot_residuals()