
# Experiments

This folder contains the baseline experiments for reference. 

## Data

Data is stored in `../data`.  We'll reserver:
- 2021 and 2022 for testing. 2022 was a very stormy year and 2021 very quiet (see `storms.md`)
- 2020 for validation
- 2000-2019 for training

In the test period we have the storms:
| Date | Storm | Peak level (Delfzijl) | Notes |
|------|-------|-----------------------|-------|
| 29–31 Jan | Corrie | — | Oosterscheldekering closed |
| ~16 Feb | Dudley | — | Part of triple storm sequence |
| 18 Feb | Eunice | +3.73 m NAP | Top 3 heaviest storms in 50+ years (KNMI); coupures Delfzijl closed; Hamburg +3.75 m surge up Elbe |
| ~21 Feb | Franklin | — | Followed immediately after Eunice |

Based on this, a zoom of February 15-25 is selected as a zoom period.

### Wind Locations
For winds we use several point sets labeled by their size:
- 9 points : hand-picked locations as a minimum for forecasting surge at main 5 locations 
    x_points = [ 3.0, 3.75, 4.25, 5.25, 6.5, 0.0,  5.0, 0.0, 0.0]
    y_points = [51.5,52.0 ,53.0 ,53.25,53.75,56.0,56.0,60.0,50.25]
### Water level locations
For the water levels, we also use multiple point sets:
- 5 main locations:
    - Vlissingen, Hoek van Holland, Den Helder, Harlingen and Delfzijl

## Tide–surge separation

Storm surge is defined as the residual after subtracting the astronomical tide:

```
surge = waterlevel − tidal_prediction
```

Tidal analysis is performed with the Schureman harmonic method (`hatyan_core`) using
94 constituents (`constituent_list("year")`). The analysis is run once on the full
2000–2022 DCSM-FM waterlevel record to obtain stable constituent estimates; the
resulting tidal prediction is then subtracted for all years.

**Script:** `analyse_tides_schureman.jl`

**Input:** `data/DCSM-FM_0_5nm_2000_2022_5stations_his.jld2`

**Output (data files):**
- `data/surge_schureman_2000_2022_5stations.jld2` — surge time series (JLD2, quantity=`"surge"`)
- `data/tides_schureman_2000_2022_5stations.jld2` — tidal prediction (JLD2, quantity=`"waterlevel"`)

**Output (diagnostics in `output_tides_schureman/`):**
- `constituents_schureman.csv` — amplitude and phase per constituent per station
- `statistics_schureman.csv` — RMSE, correlation, surge std per station
- `<station>_zoom.png` — Jan 2012 zoom plot (waterlevel, tide, surge) for each station

**Note on constituent set:** 94 constituents is the standard `"year"` set, suited to
datasets of ≥ 1 year. It is used consistently for all three training spans (1yr, 5yr,
20yr) so that baseline differences reflect only the amount of training data, not the
tidal separation method.

