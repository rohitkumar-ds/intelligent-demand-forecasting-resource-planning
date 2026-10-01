"""Fallback synthetic generator. Never overwrites an existing dataset."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd 

OUTPUT = Path(__file__).resolve().parent / "demand_data.csv"

class SyntheticDemandGenerator :
    def __init__(self,seed : int = 42) :
        self.seed = seed

    def generate(self,periods : int = 2000 , start_date : str = "2021-01-01",
                 save_path : str | Path | None = None ,overwrite : bool = False) :
        if periods < 365 :
            raise ValueError("periods must be at least 365.")
        path = Path(save_path) if save_path else OUTPUT
        if path.exists() and not overwrite :
            return pd.read_csv(path , parse_dates=["date"])
        rng = np.random.default_rng(self.seed)
        dates = pd.date_range(start_date,periods=periods,freq="D")
        t = np.arange(periods)
        dow = dates.dayofweek.to_numpy()
        doy = dates.dayofyear.to_numpy()
        annual = 1 + 0.10*np.sin(2*np.pi*(doy-1)/365.25) + 0.035*np.cos(4*np.pi*(doy-1)/365.25)
        weekly = 1 + np.where(dow == 0 , .04 ,0) + np.where(dow == 5 ,.11 ,0) + np.where(dow == 6,.16,0)
        trend = 1 + 0.00022*t
        price = 30 + 1.5*np.sin(2*np.pi*t/240) + rng.normal(0,.45,periods)
        promotion = rng.binomial(1,.095,periods)
        holiday = ((dates.month == 12)&(dates.day >=15)).astype(int) | ((dates.month==1)&(dates.day<=3)).astype(int)
        temperature = 22 + 9*np.sin(2*np.pi*(doy-80)/365.25) + rng.normal(0,2.2,periods)
        competitor_index = np.clip(100 + 5*np.sin(2*np.pi*t/180) + rng.normal(0,4,periods)-promotion*2,88,115)
        lead_time = np.clip(4.2 + 1.5 *promotion + rng.normal(0,.55,periods),2,9)

        shock = np.zeros(periods)
        shock_flags = rng.random(periods) < .012
        shock[shock_flags] = rng.choice([-1,1],shock_flags.sum()) * rng.uniform(.12,.28,shock_flags.sum())
        base_demand = 780 * trend * annual *weekly
        demand_mean = base_demand*(1+ .22*promotion + .12*holiday + 0.004*(temperature-22)
                                   - .018*(price-30) - .012*(competitor_index-100) + shock)
        demand = np.maximum(80,demand_mean + rng.normal(0,38,periods)).round().astype(int)

        asp = np.maximum(18,price + promotion*rng.normal(-1.4 , .25 , periods))
        revenue = np.maximum(0,demand*asp +rng.normal(0,260,periods))

        rolling_std = pd.Series(demand).rolling(28 , min_periods=1).std().bfill().to_numpy()
        inventory = demand*lead_time + 1.65 *rolling_std*np.sqrt(lead_time) + rng.normal(0,55,periods)
        inventory = np.maximum(0,inventory)

        staffing = np.maximum(4,3.5 + demand/190 + 1.8*promotion + rng.normal(0,.25 ,periods))
        available_capacity = np.clip(1450 + 45*np.sin(2*np.pi*t/365.25) + rng.normal(0,35,periods),1200,1600)
        utilization = np.clip((demand*1.08/available_capacity)*100 + promotion *3 + rng.normal(0,1.8,periods),15,99)
        df = pd.DataFrame({
                "date": dates,
                "product_demand": demand,
                "inventory_requirement": inventory.round(0),
                "sales_revenue": revenue.round(2),
                "staffing_requirement": staffing.round(2),
                "resource_utilization": utilization.round(2),
                "price": asp.round(2),
                "promotion": promotion.astype(int),
                "holiday": holiday.astype(int),
                "temperature": temperature.round(2),
                "competitor_index": competitor_index.round(2),
                "lead_time_days": lead_time.round(2),
            })
        if path.exists() and not overwrite:
                raise FileExistsError(f"Refusing to overwrite existing dataset: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return df

if __name__ == "__main__":
    if OUTPUT.exists():
        print(f"Dataset already exists; not overwriting: {OUTPUT}")
    else:
        df = SyntheticDemandGenerator().generate()
        print(f"Generated {len(df):,} rows -> {OUTPUT}")

        
