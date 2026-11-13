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
        session.load(telemetry=True, weather=False, messages=True)

        file_label = ''
        driver_laps = session.laps.pick_drivers(driver)

        if driver_laps.empty:
            print(f'[bold yellow]Warning: No laps found for {driver} in {session_name}. Skipping.[/bold yellow]')
            return

        if session_name in ['FP1', 'FP2', 'FP3', 'Q']:
            valid_laps = driver_laps[
                (driver_laps['IsAccurate'] == True) &
                (driver_laps['Deleted'] == False) &
                (driver_laps['TrackStatus'] == '1')
            ]

            if valid_laps.empty:
                print(f'[bold yellow]Tier 1 Validation failed for {session_name}.[/bold yellow] Applying Tier 2 Relaxed...')
                file_label = '_Tier2_Relaxed'
                valid_laps = driver_laps[
                                (driver_laps['IsAccurate'] == True) &
                                (driver_laps['Deleted'] == False)
                            ]  

            if valid_laps.empty:
                print(f'[bold yellow]Tier 2 Validation failed for {session_name}.[/bold yellow] Applying Tier 3 Fallback...')
                file_label = '_Tier3_Fallback'
                valid_laps = driver_laps[
                                driver_laps['PitOutTime'].isnull() &
                                driver_laps['PitInTime'].isnull()
                            ]
        else:
            valid_laps = driver_laps

        telemetry = valid_laps.get_telemetry()

        df = pd.DataFrame(telemetry)

        filename = f'{year}_{track}_{driver}_{session_name}{file_label}.csv'
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