from datetime import date
import json
week_starts = {1: date(2025,12,7).isoformat()}
print(week_starts)
with open('etl/configs/week_starts.json', 'w') as file:
    json.dump(week_starts, file, indent=4)