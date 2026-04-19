from apscheduler.schedulers.background import BackgroundScheduler
from node1_forecaster import retrain   # ← import retrain function from node1_forecaster.py
import time

scheduler = BackgroundScheduler()

# Retrains model every 1 minute (dev mode)
scheduler.add_job(retrain, "interval", minutes=1)
scheduler.start()

print("Scheduler running — retrains every 1 min. Press Ctrl+C to stop.")

try:
    while True:
        time.sleep(1)       # keeps the script alive
except KeyboardInterrupt:
    scheduler.shutdown()
    print("Scheduler stopped.")