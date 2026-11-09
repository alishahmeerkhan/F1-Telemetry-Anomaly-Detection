import os, fastf1
import pandas as pd
from rich import print

YEAR = 2026
TRACK = 'Monza'
DRIVER = 'LEC'
SESSIONS = ['FP1', 'FP2', 'FP3', 'Q', 'R']


RAW_DATA_DIR = 'data/raw'
CACHE_DIR = 'data/cache'

os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

fastf1.Cache.enable_cache(cache_dir=CACHE_DIR)

def fetch_data(year, track, driver, session_name):
    print(f'Fetching [bold cyan]{year} {track} - {session_name} for {driver}.[/bold cyan]')

    try:
        session = fastf1.get_session(year, track, session_name)
        session.load(telemetry=True, weather=False, messages=False)

        driver_laps = session.laps.pick_driver(driver)

        if driver_laps.empty:
            print(f'[bold yellow]Warning: No laps found for {driver} in {session_name}. Skipping.[/bold yellow]')
            return

        telemetry = driver_laps.get_telemetry()

        df = pd.DataFrame(telemetry)

        filename = f'{year}_{track}_{driver}_{session_name}.csv'
        file_path = os.path.join(RAW_DATA_DIR, filename)

        df.to_csv(file_path, index=False)
        print(f'[bold green]Success: Saved {len(df)} rows to {file_path}[bold green]')

    except Exception as e:
        print(f'[bold red]Error fetching {session_name}:[/bold red] {e}')

if __name__ == '__main__':
    print('Loading All Practice sessions, Qualifying and Race data data for ' \
    f'[bold cyan]Charles Leclerc at {TRACK} {YEAR}.[/bold cyan]')

    for session in SESSIONS:
        fetch_data(year=YEAR, track=TRACK, driver=DRIVER, session_name=session)