# On-demand feedback
## Article
> **Functional improvement in β cell models of type 2 diabetes using on-demand feedback control**
> Keita Koyama, Hiroyasu Ando, Kantaro Fujiwara  
> AIP Advances Volume13, Isuue4, April 2023  
> https://doi.org/10.1063/5.0124625
## src
### python script
- numerical_solution.py
  - runge-kutta method
  - euler-maruyama method
- model.py
  - Chay model
  - Riz model
  - (Hindmarsh-Rose model)
- calculate.py
  - Supports digital-on-demand input and noise application.
- simulation_***.py
  - Parameter area survey of stimulus efficacy
  - chaotic Chay
  - stochastic Riz
- heatmap.py

### notebook
- ShowHeatmap.ipynb (using functions in hetamap.py)
- TimeSeries.ipynb

