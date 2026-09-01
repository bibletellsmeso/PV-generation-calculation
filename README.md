# PV Generation Calculation

Python scripts for photovoltaic generation calculation from weather and irradiance inputs, including preprocessing and two historical model variants.

## Contents

| File | Purpose |
|---|---|
| `Model 1.py`, `Model 2.py` | Historical photovoltaic-model variants. |
| `Preprocessing.py` | Prepares weather and scheduling inputs. |
| `Weather Forecast.csv`, `Hawaii weather.csv` | Weather input files used by the scripts. |
| `Irradiance.csv`, `Degree.csv` | Supporting irradiance and angle inputs. |
| `PV_for_scheduling.txt` | Scheduling-oriented PV series. |
| `result.png` | Preserved historical figure. |

The scripts use relative file paths and should be run from the repository root. Before using the results for a new study, confirm weather-data provenance, units, and the selected location and panel parameters in the script.

## Requirements

- Python 3.8 or later
- `numpy`, `pandas`, and `matplotlib`

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python "Preprocessing.py"
```

Then review the configuration near the top of `Model 1.py` or `Model 2.py` before running a model variant.

## Cleanup status

Existing filenames are historical. This branch documents their roles but does not rename, delete, or redistribute data. A later validated migration will use descriptive, space-free filenames and record the path mapping.

## License and citation

A reuse license and citation guidance will be added after the intended reuse terms are chosen.
