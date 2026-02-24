import pandas as pd

df5 = pd.DataFrame(Statistics.validatorTelemetry)
df5.columns = [
    "epoch",
    "validator_id",
    "vote_delay",
    "missed_vote_rate",
    "uptime",
    "connectivity_degree"
]

df5.to_excel(writer, sheet_name='ValidatorTelemetry')
