# Dataset Notes

## Source

**Flight Delay Dataset (2018–2022)**  
Kaggle: https://www.kaggle.com/datasets/robikscube/flight-delay-dataset-20182022

The project uses the **2022 flight file** from the larger multi-year collection.

## Raw 2022 Data

- Records: **4,078,318**
- Columns: **61**
- Domain: U.S. domestic flight operations
- Target used in this project: **`ArrDel15`**
- Target definition: `1` when arrival delay exceeds 15 minutes, otherwise `0`

The raw variables include scheduled and actual timing fields, airline identifiers, origin/destination information, taxi times, flight distance, elapsed time, and delay indicators.

## Exploratory Scope

The complete 2022 file was used for exploratory summaries such as:

- top origin airports;
- delay by hour;
- delay by month;
- delay by weekday;
- numerical correlations.

The ten busiest origin airports observed in the source notebook were:

| Rank | Airport | Flights |
|---:|---|---:|
| 1 | ATL | 183,697 |
| 2 | ORD | 172,169 |
| 3 | DFW | 161,218 |
| 4 | DEN | 158,092 |
| 5 | CLT | 126,088 |
| 6 | LAX | 111,935 |
| 7 | LGA | 100,260 |
| 8 | SEA | 98,884 |
| 9 | LAS | 97,674 |
| 10 | PHX | 96,095 |

## Modeling Scope

The implementation used **ATL-origin flights** as the modeling subset.

- ATL-origin rows before final preparation: **183,697**
- Prepared rows after target-related missing-value removal: **179,852**
- Prepared columns: **33**, including the target and the retained departure-delay indicator

Target counts in the prepared data:

| Class | Meaning | Count | Share |
|---:|---|---:|---:|
| 0 | arrival delay ≤ 15 min | 144,410 | 80.3% |
| 1 | arrival delay > 15 min | 35,442 | 19.7% |

## Core Raw Modeling Variables

The reduced modeling table retained 15 raw columns before time expansion and categorical encoding:

- TaxiIn
- TaxiOut
- ArrDel15
- DepDel15
- DayofMonth
- Month
- DepTime
- WheelsOff
- WheelsOn
- Distance
- CRSDepTime
- CRSArrTime
- Airline
- AirTime
- DepHour

Five HHMM timing columns were converted into sine/cosine pairs:

- CRSDepTime
- DepTime
- CRSArrTime
- WheelsOff
- WheelsOn

## Important Data-Availability Note

Several modeling variables are not necessarily known before departure. `DepDel15`, actual departure time, TaxiIn/TaxiOut, WheelsOff, and WheelsOn can become available during or after flight operations.

Therefore, the reported experiment should be interpreted as **operational delay-status classification** unless the prediction moment is explicitly defined. A future pre-departure model should remove all variables unavailable at prediction time.

## Data Distribution

The dataset is not included in this repository. Users should obtain it from the original Kaggle source and follow its licensing and usage terms.
