import json
import os
import argparse
from datetime import datetime

def extract_data(start_date, end_date, input_dir):
    # Ensure input directory exists
    if not os.path.exists(input_dir):
        raise FileNotFoundError("Directory doesn't exist")
    
    start_year, start_month = start_date.year, start_date.month
    end_year, end_month = end_date.year, end_date.month

    # Parsed entries will be added to this array
    entries = []
    
    # Loop through the months and extract lead paragraphs
    for year in range(start_year, end_year + 1):
        for month in range(start_month if year != end_year else 1, 13 if year != end_year else end_month + 1):
            fn = f"{year}_{month}.json"
            path = os.path.join(input_dir, fn)
            with open(path, "r") as file:
                print(f"Extracting lead paragraphs from {fn}")
                data = json.load(file)
                # print(data["response"]["docs"][0].keys())
                docs = data["response"]["docs"]
                for entry in docs:
                    _id = entry["_id"]
                    lead_paragraph = entry["lead_paragraph"]
                    pub_date = entry["pub_date"]
                    extracted = {"_id": _id, "lead_paragraph": lead_paragraph, "pub_date": pub_date}
                    entries.append(extracted)
    
    with open(os.path.join(input_dir, "nyt_parsed.json"), "w") as outfile:
        json.dump(entries, outfile)


def main():
    parser = argparse.ArgumentParser(description='Extract lead paragraphs from downloaded NYT Archive Data.')
    parser.add_argument('start_date', type=lambda d: datetime.strptime(d, '%Y-%m'), help='Start date in YYYY-MM format')
    parser.add_argument('end_date', type=lambda d: datetime.strptime(d, '%Y-%m'), help='End date in YYYY-MM format')
    parser.add_argument('input_dir', type=str, help='Input directory of the downloaded data')

    args = parser.parse_args()
    
    extract_data(args.start_date, args.end_date, args.input_dir)

if __name__ == '__main__':
    main()
