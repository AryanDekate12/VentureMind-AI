import pandas as pd
import numpy as np
import os

np.random.seed(42)

n = 2000

funding = np.random.randint(0, 200, n)
team_size = np.random.randint(1, 100, n)
market_size = np.random.randint(1, 500, n)
years = np.random.randint(0, 10, n)
experience = np.random.randint(0, 20, n)

# success probability formula
success_prob = (
    0.35 * (funding / 200)
    + 0.20 * (team_size / 100)
    + 0.20 * (market_size / 500)
    + 0.10 * (years / 10)
    + 0.15 * (experience / 20)
)

success = (success_prob > 0.5).astype(int)

df = pd.DataFrame({
    "funding_million": funding,
    "team_size": team_size,
    "market_size_billion": market_size,
    "years_operating": years,
    "founder_experience": experience,
    "success": success
})

os.makedirs("dataset", exist_ok=True)

df.to_csv("dataset/startup_data.csv", index=False)

print("Dataset generated successfully")